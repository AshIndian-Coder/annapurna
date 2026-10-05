import '../../core/api/api_client.dart';

/// Result of uploading a footfall / orders history file.
class DatasetUploadResult {
  final String datasetId;
  final String filename;
  final int rowsImported;
  final int rowsRejected;
  final List<String> columnsFound;
  final List<String> rejectSamples;
  final String? earliestDate;
  final String? latestDate;
  final int dateRangeDays;

  const DatasetUploadResult({
    required this.datasetId,
    required this.filename,
    required this.rowsImported,
    required this.rowsRejected,
    required this.columnsFound,
    required this.rejectSamples,
    this.earliestDate,
    this.latestDate,
    required this.dateRangeDays,
  });

  factory DatasetUploadResult.fromJson(Map<String, dynamic> json) => DatasetUploadResult(
    datasetId: json['dataset_id'] as String? ?? '',
    filename: json['filename'] as String? ?? '',
    rowsImported: (json['rows_imported'] as num?)?.toInt() ?? 0,
    rowsRejected: (json['rows_rejected'] as num?)?.toInt() ?? 0,
    columnsFound: (json['columns_found'] as List?)?.cast<String>() ?? const [],
    rejectSamples: (json['reject_samples'] as List?)?.cast<String>() ?? const [],
    earliestDate: json['earliest_date'] as String?,
    latestDate: json['latest_date'] as String?,
    dateRangeDays: (json['date_range_days'] as num?)?.toInt() ?? 0,
  );

  bool get hasRejectedRows => rowsRejected > 0;
}

/// A previously uploaded dataset.
class DatasetInfo {
  final String datasetId;
  final String filename;
  final int rowsImported;
  final int rowsRejected;
  final List<String> columnsFound;
  final String? earliestDate;
  final String? latestDate;

  const DatasetInfo({
    required this.datasetId,
    required this.filename,
    required this.rowsImported,
    required this.rowsRejected,
    required this.columnsFound,
    this.earliestDate,
    this.latestDate,
  });

  factory DatasetInfo.fromJson(Map<String, dynamic> json) => DatasetInfo(
    datasetId: json['dataset_id'] as String? ?? '',
    filename: json['filename'] as String? ?? '',
    rowsImported: (json['rows_imported'] as num?)?.toInt() ?? 0,
    rowsRejected: (json['rows_rejected'] as num?)?.toInt() ?? 0,
    columnsFound: (json['columns_found'] as List?)?.cast<String>() ?? const [],
    earliestDate: json['earliest_date'] as String?,
    latestDate: json['latest_date'] as String?,
  );
}

/// Small helper so the DTO file does not need the Result import.
List<DatasetInfo> parseDatasetList(dynamic data) =>
    extractList<DatasetInfo>(data, DatasetInfo.fromJson);