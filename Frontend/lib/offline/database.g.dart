// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'database.dart';

// ignore_for_file: type=lint
class $OutboxItemsTable extends OutboxItems
    with TableInfo<$OutboxItemsTable, OutboxItem> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $OutboxItemsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _clientEventIdMeta =
      const VerificationMeta('clientEventId');
  @override
  late final GeneratedColumn<String> clientEventId = GeneratedColumn<String>(
      'client_event_id', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _kindMeta = const VerificationMeta('kind');
  @override
  late final GeneratedColumn<String> kind = GeneratedColumn<String>(
      'kind', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _payloadMeta =
      const VerificationMeta('payload');
  @override
  late final GeneratedColumn<String> payload = GeneratedColumn<String>(
      'payload', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _dependsOnMeta =
      const VerificationMeta('dependsOn');
  @override
  late final GeneratedColumn<String> dependsOn = GeneratedColumn<String>(
      'depends_on', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _createdAtMeta =
      const VerificationMeta('createdAt');
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
      'created_at', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  static const VerificationMeta _stateMeta = const VerificationMeta('state');
  @override
  late final GeneratedColumn<String> state = GeneratedColumn<String>(
      'state', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant('QUEUED'));
  static const VerificationMeta _retryCountMeta =
      const VerificationMeta('retryCount');
  @override
  late final GeneratedColumn<int> retryCount = GeneratedColumn<int>(
      'retry_count', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultValue: const Constant(0));
  @override
  List<GeneratedColumn> get $columns =>
      [clientEventId, kind, payload, dependsOn, createdAt, state, retryCount];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'outbox_items';
  @override
  VerificationContext validateIntegrity(Insertable<OutboxItem> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('client_event_id')) {
      context.handle(
          _clientEventIdMeta,
          clientEventId.isAcceptableOrUnknown(
              data['client_event_id']!, _clientEventIdMeta));
    } else if (isInserting) {
      context.missing(_clientEventIdMeta);
    }
    if (data.containsKey('kind')) {
      context.handle(
          _kindMeta, kind.isAcceptableOrUnknown(data['kind']!, _kindMeta));
    } else if (isInserting) {
      context.missing(_kindMeta);
    }
    if (data.containsKey('payload')) {
      context.handle(_payloadMeta,
          payload.isAcceptableOrUnknown(data['payload']!, _payloadMeta));
    } else if (isInserting) {
      context.missing(_payloadMeta);
    }
    if (data.containsKey('depends_on')) {
      context.handle(_dependsOnMeta,
          dependsOn.isAcceptableOrUnknown(data['depends_on']!, _dependsOnMeta));
    }
    if (data.containsKey('created_at')) {
      context.handle(_createdAtMeta,
          createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta));
    } else if (isInserting) {
      context.missing(_createdAtMeta);
    }
    if (data.containsKey('state')) {
      context.handle(
          _stateMeta, state.isAcceptableOrUnknown(data['state']!, _stateMeta));
    }
    if (data.containsKey('retry_count')) {
      context.handle(
          _retryCountMeta,
          retryCount.isAcceptableOrUnknown(
              data['retry_count']!, _retryCountMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {clientEventId};
  @override
  OutboxItem map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return OutboxItem(
      clientEventId: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}client_event_id'])!,
      kind: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}kind'])!,
      payload: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}payload'])!,
      dependsOn: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}depends_on']),
      createdAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}created_at'])!,
      state: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}state'])!,
      retryCount: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}retry_count'])!,
    );
  }

  @override
  $OutboxItemsTable createAlias(String alias) {
    return $OutboxItemsTable(attachedDatabase, alias);
  }
}

class OutboxItem extends DataClass implements Insertable<OutboxItem> {
  final String clientEventId;
  final String kind;
  final String payload;
  final String? dependsOn;
  final DateTime createdAt;
  final String state;
  final int retryCount;
  const OutboxItem(
      {required this.clientEventId,
      required this.kind,
      required this.payload,
      this.dependsOn,
      required this.createdAt,
      required this.state,
      required this.retryCount});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['client_event_id'] = Variable<String>(clientEventId);
    map['kind'] = Variable<String>(kind);
    map['payload'] = Variable<String>(payload);
    if (!nullToAbsent || dependsOn != null) {
      map['depends_on'] = Variable<String>(dependsOn);
    }
    map['created_at'] = Variable<DateTime>(createdAt);
    map['state'] = Variable<String>(state);
    map['retry_count'] = Variable<int>(retryCount);
    return map;
  }

  OutboxItemsCompanion toCompanion(bool nullToAbsent) {
    return OutboxItemsCompanion(
      clientEventId: Value(clientEventId),
      kind: Value(kind),
      payload: Value(payload),
      dependsOn: dependsOn == null && nullToAbsent
          ? const Value.absent()
          : Value(dependsOn),
      createdAt: Value(createdAt),
      state: Value(state),
      retryCount: Value(retryCount),
    );
  }

  factory OutboxItem.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return OutboxItem(
      clientEventId: serializer.fromJson<String>(json['clientEventId']),
      kind: serializer.fromJson<String>(json['kind']),
      payload: serializer.fromJson<String>(json['payload']),
      dependsOn: serializer.fromJson<String?>(json['dependsOn']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
      state: serializer.fromJson<String>(json['state']),
      retryCount: serializer.fromJson<int>(json['retryCount']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'clientEventId': serializer.toJson<String>(clientEventId),
      'kind': serializer.toJson<String>(kind),
      'payload': serializer.toJson<String>(payload),
      'dependsOn': serializer.toJson<String?>(dependsOn),
      'createdAt': serializer.toJson<DateTime>(createdAt),
      'state': serializer.toJson<String>(state),
      'retryCount': serializer.toJson<int>(retryCount),
    };
  }

  OutboxItem copyWith(
          {String? clientEventId,
          String? kind,
          String? payload,
          Value<String?> dependsOn = const Value.absent(),
          DateTime? createdAt,
          String? state,
          int? retryCount}) =>
      OutboxItem(
        clientEventId: clientEventId ?? this.clientEventId,
        kind: kind ?? this.kind,
        payload: payload ?? this.payload,
        dependsOn: dependsOn.present ? dependsOn.value : this.dependsOn,
        createdAt: createdAt ?? this.createdAt,
        state: state ?? this.state,
        retryCount: retryCount ?? this.retryCount,
      );
  OutboxItem copyWithCompanion(OutboxItemsCompanion data) {
    return OutboxItem(
      clientEventId: data.clientEventId.present
          ? data.clientEventId.value
          : this.clientEventId,
      kind: data.kind.present ? data.kind.value : this.kind,
      payload: data.payload.present ? data.payload.value : this.payload,
      dependsOn: data.dependsOn.present ? data.dependsOn.value : this.dependsOn,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
      state: data.state.present ? data.state.value : this.state,
      retryCount:
          data.retryCount.present ? data.retryCount.value : this.retryCount,
    );
  }

  @override
  String toString() {
    return (StringBuffer('OutboxItem(')
          ..write('clientEventId: $clientEventId, ')
          ..write('kind: $kind, ')
          ..write('payload: $payload, ')
          ..write('dependsOn: $dependsOn, ')
          ..write('createdAt: $createdAt, ')
          ..write('state: $state, ')
          ..write('retryCount: $retryCount')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
      clientEventId, kind, payload, dependsOn, createdAt, state, retryCount);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is OutboxItem &&
          other.clientEventId == this.clientEventId &&
          other.kind == this.kind &&
          other.payload == this.payload &&
          other.dependsOn == this.dependsOn &&
          other.createdAt == this.createdAt &&
          other.state == this.state &&
          other.retryCount == this.retryCount);
}

class OutboxItemsCompanion extends UpdateCompanion<OutboxItem> {
  final Value<String> clientEventId;
  final Value<String> kind;
  final Value<String> payload;
  final Value<String?> dependsOn;
  final Value<DateTime> createdAt;
  final Value<String> state;
  final Value<int> retryCount;
  final Value<int> rowid;
  const OutboxItemsCompanion({
    this.clientEventId = const Value.absent(),
    this.kind = const Value.absent(),
    this.payload = const Value.absent(),
    this.dependsOn = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.state = const Value.absent(),
    this.retryCount = const Value.absent(),
    this.rowid = const Value.absent(),
  });
  OutboxItemsCompanion.insert({
    required String clientEventId,
    required String kind,
    required String payload,
    this.dependsOn = const Value.absent(),
    required DateTime createdAt,
    this.state = const Value.absent(),
    this.retryCount = const Value.absent(),
    this.rowid = const Value.absent(),
  })  : clientEventId = Value(clientEventId),
        kind = Value(kind),
        payload = Value(payload),
        createdAt = Value(createdAt);
  static Insertable<OutboxItem> custom({
    Expression<String>? clientEventId,
    Expression<String>? kind,
    Expression<String>? payload,
    Expression<String>? dependsOn,
    Expression<DateTime>? createdAt,
    Expression<String>? state,
    Expression<int>? retryCount,
    Expression<int>? rowid,
  }) {
    return RawValuesInsertable({
      if (clientEventId != null) 'client_event_id': clientEventId,
      if (kind != null) 'kind': kind,
      if (payload != null) 'payload': payload,
      if (dependsOn != null) 'depends_on': dependsOn,
      if (createdAt != null) 'created_at': createdAt,
      if (state != null) 'state': state,
      if (retryCount != null) 'retry_count': retryCount,
      if (rowid != null) 'rowid': rowid,
    });
  }

  OutboxItemsCompanion copyWith(
      {Value<String>? clientEventId,
      Value<String>? kind,
      Value<String>? payload,
      Value<String?>? dependsOn,
      Value<DateTime>? createdAt,
      Value<String>? state,
      Value<int>? retryCount,
      Value<int>? rowid}) {
    return OutboxItemsCompanion(
      clientEventId: clientEventId ?? this.clientEventId,
      kind: kind ?? this.kind,
      payload: payload ?? this.payload,
      dependsOn: dependsOn ?? this.dependsOn,
      createdAt: createdAt ?? this.createdAt,
      state: state ?? this.state,
      retryCount: retryCount ?? this.retryCount,
      rowid: rowid ?? this.rowid,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (clientEventId.present) {
      map['client_event_id'] = Variable<String>(clientEventId.value);
    }
    if (kind.present) {
      map['kind'] = Variable<String>(kind.value);
    }
    if (payload.present) {
      map['payload'] = Variable<String>(payload.value);
    }
    if (dependsOn.present) {
      map['depends_on'] = Variable<String>(dependsOn.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    if (state.present) {
      map['state'] = Variable<String>(state.value);
    }
    if (retryCount.present) {
      map['retry_count'] = Variable<int>(retryCount.value);
    }
    if (rowid.present) {
      map['rowid'] = Variable<int>(rowid.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('OutboxItemsCompanion(')
          ..write('clientEventId: $clientEventId, ')
          ..write('kind: $kind, ')
          ..write('payload: $payload, ')
          ..write('dependsOn: $dependsOn, ')
          ..write('createdAt: $createdAt, ')
          ..write('state: $state, ')
          ..write('retryCount: $retryCount, ')
          ..write('rowid: $rowid')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $OutboxItemsTable outboxItems = $OutboxItemsTable(this);
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities => [outboxItems];
}

typedef $$OutboxItemsTableCreateCompanionBuilder = OutboxItemsCompanion
    Function({
  required String clientEventId,
  required String kind,
  required String payload,
  Value<String?> dependsOn,
  required DateTime createdAt,
  Value<String> state,
  Value<int> retryCount,
  Value<int> rowid,
});
typedef $$OutboxItemsTableUpdateCompanionBuilder = OutboxItemsCompanion
    Function({
  Value<String> clientEventId,
  Value<String> kind,
  Value<String> payload,
  Value<String?> dependsOn,
  Value<DateTime> createdAt,
  Value<String> state,
  Value<int> retryCount,
  Value<int> rowid,
});

class $$OutboxItemsTableFilterComposer
    extends Composer<_$AppDatabase, $OutboxItemsTable> {
  $$OutboxItemsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<String> get clientEventId => $composableBuilder(
      column: $table.clientEventId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get kind => $composableBuilder(
      column: $table.kind, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get payload => $composableBuilder(
      column: $table.payload, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get dependsOn => $composableBuilder(
      column: $table.dependsOn, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get state => $composableBuilder(
      column: $table.state, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => ColumnFilters(column));
}

class $$OutboxItemsTableOrderingComposer
    extends Composer<_$AppDatabase, $OutboxItemsTable> {
  $$OutboxItemsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<String> get clientEventId => $composableBuilder(
      column: $table.clientEventId,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get kind => $composableBuilder(
      column: $table.kind, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get payload => $composableBuilder(
      column: $table.payload, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get dependsOn => $composableBuilder(
      column: $table.dependsOn, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get state => $composableBuilder(
      column: $table.state, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => ColumnOrderings(column));
}

class $$OutboxItemsTableAnnotationComposer
    extends Composer<_$AppDatabase, $OutboxItemsTable> {
  $$OutboxItemsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<String> get clientEventId => $composableBuilder(
      column: $table.clientEventId, builder: (column) => column);

  GeneratedColumn<String> get kind =>
      $composableBuilder(column: $table.kind, builder: (column) => column);

  GeneratedColumn<String> get payload =>
      $composableBuilder(column: $table.payload, builder: (column) => column);

  GeneratedColumn<String> get dependsOn =>
      $composableBuilder(column: $table.dependsOn, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  GeneratedColumn<String> get state =>
      $composableBuilder(column: $table.state, builder: (column) => column);

  GeneratedColumn<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => column);
}

class $$OutboxItemsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $OutboxItemsTable,
    OutboxItem,
    $$OutboxItemsTableFilterComposer,
    $$OutboxItemsTableOrderingComposer,
    $$OutboxItemsTableAnnotationComposer,
    $$OutboxItemsTableCreateCompanionBuilder,
    $$OutboxItemsTableUpdateCompanionBuilder,
    (OutboxItem, BaseReferences<_$AppDatabase, $OutboxItemsTable, OutboxItem>),
    OutboxItem,
    PrefetchHooks Function()> {
  $$OutboxItemsTableTableManager(_$AppDatabase db, $OutboxItemsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$OutboxItemsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$OutboxItemsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$OutboxItemsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<String> clientEventId = const Value.absent(),
            Value<String> kind = const Value.absent(),
            Value<String> payload = const Value.absent(),
            Value<String?> dependsOn = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<String> state = const Value.absent(),
            Value<int> retryCount = const Value.absent(),
            Value<int> rowid = const Value.absent(),
          }) =>
              OutboxItemsCompanion(
            clientEventId: clientEventId,
            kind: kind,
            payload: payload,
            dependsOn: dependsOn,
            createdAt: createdAt,
            state: state,
            retryCount: retryCount,
            rowid: rowid,
          ),
          createCompanionCallback: ({
            required String clientEventId,
            required String kind,
            required String payload,
            Value<String?> dependsOn = const Value.absent(),
            required DateTime createdAt,
            Value<String> state = const Value.absent(),
            Value<int> retryCount = const Value.absent(),
            Value<int> rowid = const Value.absent(),
          }) =>
              OutboxItemsCompanion.insert(
            clientEventId: clientEventId,
            kind: kind,
            payload: payload,
            dependsOn: dependsOn,
            createdAt: createdAt,
            state: state,
            retryCount: retryCount,
            rowid: rowid,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ));
}

typedef $$OutboxItemsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $OutboxItemsTable,
    OutboxItem,
    $$OutboxItemsTableFilterComposer,
    $$OutboxItemsTableOrderingComposer,
    $$OutboxItemsTableAnnotationComposer,
    $$OutboxItemsTableCreateCompanionBuilder,
    $$OutboxItemsTableUpdateCompanionBuilder,
    (OutboxItem, BaseReferences<_$AppDatabase, $OutboxItemsTable, OutboxItem>),
    OutboxItem,
    PrefetchHooks Function()>;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$OutboxItemsTableTableManager get outboxItems =>
      $$OutboxItemsTableTableManager(_db, _db.outboxItems);
}
