import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'dart:io';

part 'database.g.dart';

class OutboxItems extends Table {
  TextColumn get clientEventId => text()();
  TextColumn get kind => text()();
  TextColumn get payload => text()();
  TextColumn get dependsOn => text().nullable()();
  DateTimeColumn get createdAt => dateTime()();
  TextColumn get state => text().withDefault(const Constant('QUEUED'))();
  IntColumn get retryCount => integer().withDefault(const Constant(0))();

  @override
  Set<Column> get primaryKey => {clientEventId};
}

@DriftDatabase(tables: [OutboxItems])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  @override
  int get schemaVersion => 1;

  Future<void> insertOutboxItem(OutboxItem item) => into(outboxItems).insert(item, mode: InsertMode.replace);
  
  Future<List<OutboxItem>> getPendingItems() => 
    (select(outboxItems)
      ..where((t) => t.state.isIn(['QUEUED', 'SENDING', 'FAILED']))
      ..where((t) => t.retryCount.isSmallerThanValue(5))
      ..orderBy([(t) => OrderingTerm(expression: t.createdAt, mode: OrderingMode.asc)])
    ).get();

  Future<void> updateItemState(String id, String newState, int newRetryCount) {
    return (update(outboxItems)..where((t) => t.clientEventId.equals(id))).write(
      OutboxItemsCompanion(
        state: Value(newState),
        retryCount: Value(newRetryCount),
      ),
    );
  }
}

LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    final dbFolder = await getApplicationDocumentsDirectory();
    final file = File(p.join(dbFolder.path, 'annapurna_offline.sqlite'));
    return NativeDatabase.createInBackground(file);
  });
}
