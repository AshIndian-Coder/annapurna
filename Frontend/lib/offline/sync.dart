import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'dart:convert';
import 'database.dart';
import 'outbox.dart';
import '../core/api/api_client.dart';

final syncManagerProvider = Provider<SyncManager>((ref) {
  return SyncManager(ref.read(databaseProvider), ref.read(apiClientProvider));
});

class SyncManager {
  final AppDatabase _db;
  final ApiClient _api;
  bool _isSyncing = false;

  SyncManager(this._db, this._api);

  Future<void> syncOutbox() async {
    if (_isSyncing) return;
    _isSyncing = true;
    try {
      final items = await _db.getPendingItems();
      if (items.isEmpty) return;

      for (var item in items) {
         await _db.updateItemState(item.clientEventId, 'SENDING', item.retryCount);
      }

      final payload = {
        'items': items.map((i) => {
          'client_event_id': i.clientEventId,
          'kind': i.kind,
          'payload': jsonDecode(i.payload),
          'client_ts': i.createdAt.toIso8601String(),
        }).toList()
      };

      final response = await _api.post('/sync/batch', data: payload, parser: (data) => extractList<Map<String, dynamic>>(data, (m) => m));

      await response.when(
        success: (results) async {
          for (var res in results) {
            final id = res['client_event_id'] as String;
            final status = res['status'] as String;
            
            if (status == 'ACCEPTED' || status == 'DUPLICATE') {
              await _db.updateItemState(id, 'DONE', 0);
            } else {
              final original = items.firstWhere((i) => i.clientEventId == id);
              final newRetries = original.retryCount + 1;
              await _db.updateItemState(id, newRetries >= 5 ? 'DEAD' : 'FAILED', newRetries);
            }
          }
        },
        failure: (err) async {
          for (var item in items) {
             final newRetries = item.retryCount + 1;
             await _db.updateItemState(item.clientEventId, newRetries >= 5 ? 'DEAD' : 'FAILED', newRetries);
          }
        }
      );
    } finally {
      _isSyncing = false;
    }
  }
}
