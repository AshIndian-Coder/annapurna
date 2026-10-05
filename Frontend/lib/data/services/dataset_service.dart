import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/dataset.dart';

final datasetServiceProvider = Provider<DatasetService>((ref) {
  return DatasetService(ref.read(apiClientProvider));
});

/// File extensions the backend accepts for a history upload.
const supportedDatasetExtensions = ['csv', 'tsv', 'txt', 'xlsx', 'xlsm'];

class DatasetService {
  final ApiClient _api;
  DatasetService(this._api);

  /// Opens the system file picker restricted to the formats the backend reads.
  ///
  /// Returns the chosen path, or null when the user cancels.
  Future<PickedDatasetFile?> pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: supportedDatasetExtensions,
      withData: true, // read in memory so the upload works on every platform
    );
    if (result == null || result.files.isEmpty) return null;

    final picked = result.files.first;
    if (picked.bytes == null) {
      return PickedDatasetFile(name: picked.name, path: null, bytes: null);
    }
    return PickedDatasetFile(name: picked.name, path: picked.path, bytes: picked.bytes!);
  }

  /// Uploads a history file to POST /kitchen/datasets.
  Future<Result<DatasetUploadResult>> upload(PickedDatasetFile file) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(
        file.bytes!,
        filename: file.name,
      ),
    });

    return _api.postMultipart<DatasetUploadResult>(
      '/kitchen/datasets',
      formData: form,
      parser: (data) => DatasetUploadResult.fromJson(extractMap(data)),
    );
  }

  /// Lists the kitchen's previously uploaded datasets.
  Future<Result<List<DatasetInfo>>> list() async {
    return _api.get<List<DatasetInfo>>(
      '/kitchen/datasets',
      parser: (data) => parseDatasetList(data),
    );
  }
}

class PickedDatasetFile {
  final String name;
  final String? path;
  final List<int>? bytes;

  const PickedDatasetFile({required this.name, this.path, this.bytes});

  String get extension {
    final dot = name.lastIndexOf('.');
    return dot < 0 ? '' : name.substring(dot + 1).toLowerCase();
  }
}