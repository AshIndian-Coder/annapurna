import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import 'dart:convert';
import 'database.dart';

final databaseProvider = Provider<AppDatabase>((ref) {
  return AppDatabase();
});

final outboxServiceProvider = Provider<OutboxService>((ref) {
  return OutboxService(ref.read(databaseProvider));
});

class OutboxService {
  final AppDatabase _db;
  OutboxService(this._db);

  Future<String> enqueueMutation({
    required String kind,
    required Map<String, dynamic> payload,
    String? dependsOn,
  }) async {
    final eventId = payload['client_event_id'] ?? const Uuid().v7();
    payload['client_event_id'] = eventId;
    payload['client_ts'] = DateTime.now().toIso8601String();

    await _db.insertOutboxItem(OutboxItem(
      clientEventId: eventId,
      kind: kind,
      payload: jsonEncode(payload),
      dependsOn: dependsOn,
      createdAt: DateTime.now(),
      state: 'QUEUED',
      retryCount: 0,
    ));

    return eventId;
  }
}
