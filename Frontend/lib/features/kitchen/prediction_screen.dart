import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../data/dtos/dataset.dart';
import '../../data/dtos/models.dart';
import '../../data/services/dataset_service.dart';
import '../../data/services/kitchen_service.dart';
import '../../shared/widgets/common_widgets.dart';

/// Demand prediction for a kitchen.
///
/// The prediction itself is produced by the backend, which forwards the feature
/// vector to the ML service. This screen's job is to let the kitchen upload the
/// historical footfall / orders data that makes the prediction accurate.
class PredictionScreen extends ConsumerStatefulWidget {
  const PredictionScreen({super.key});

  @override
  ConsumerState<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends ConsumerState<PredictionScreen> {
  final _attendanceController = TextEditingController();

  MealType _selectedMeal = MealType.lunch;
  final List<String> _selectedMenu = ['rice', 'dal', 'paneer'];

  PickedDatasetFile? _pendingFile;
  DatasetUploadResult? _uploaded;
  List<DatasetInfo> _datasets = const [];
  String? _uploadError;

  bool _uploading = false;
  bool _predicting = false;
  Prediction? _result;

  static const _menuOptions = [
    'rice', 'dal', 'paneer', 'roti', 'sabji', 'biryani', 'curd', 'salad', 'soup', 'sweet',
  ];

  @override
  void initState() {
    super.initState();
    _loadDatasets();
  }

  @override
  void dispose() {
    _attendanceController.dispose();
    super.dispose();
  }

  Future<void> _loadDatasets() async {
    final result = await ref.read(datasetServiceProvider).list();
    if (!mounted) return;
    result.when(
      success: (data) => setState(() => _datasets = data),
      failure: (_) => setState(() => _datasets = const []),
    );
  }

  Future<void> _chooseFile() async {
    setState(() => _uploadError = null);
    try {
      final picked = await ref.read(datasetServiceProvider).pickFile();
      if (picked == null || !mounted) return;
      setState(() {
        _pendingFile = picked;
        _uploaded = null;
      });
    } catch (e) {
      if (mounted) setState(() => _uploadError = 'Could not open the file picker: $e');
    }
  }

  Future<void> _upload() async {
    final file = _pendingFile;
    if (file == null) return;
    if (file.bytes == null) {
      setState(() => _uploadError = 'That file could not be read.');
      return;
    }

    setState(() {
      _uploading = true;
      _uploadError = null;
    });

    final result = await ref.read(datasetServiceProvider).upload(file);
    if (!mounted) return;

    result.when(
      success: (data) {
        setState(() {
          _uploading = false;
          _uploaded = data;
          _pendingFile = null;
        });
        _loadDatasets();
      },
      failure: (error) => setState(() {
        _uploading = false;
        _uploadError = error.message;
      }),
    );
  }

  Future<void> _predict() async {
    setState(() => _predicting = true);
    final now = DateTime.now();
    final result = await ref.read(kitchenServiceProvider).predictDemand(
      attendance: int.tryParse(_attendanceController.text) ?? 0,
      mealType: _selectedMeal.apiValue,
      menu: _selectedMenu,
      dayOfWeek: now.weekday - 1,
      date: '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}',
      datasetId: _uploaded?.datasetId ?? (_datasets.isNotEmpty ? _datasets.first.datasetId : null),
    );
    if (!mounted) return;
    result.when(
      success: (data) => setState(() {
        _predicting = false;
        _result = data;
      }),
      failure: (error) => setState(() {
        _predicting = false;
        _uploadError = error.message;
      }),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Demand Prediction')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildUploadCard(),
          const SizedBox(height: 16),
          _buildInputCard(),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton.icon(
              onPressed: _predicting ? null : _predict,
              icon: _predicting
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.auto_graph),
              label: Text(_predicting ? 'Predicting...' : 'Predict Demand'),
            ),
          ),
          if (_result != null) ...[
            const SizedBox(height: 24),
            _buildResultCard(_result!),
            const SizedBox(height: 16),
            _buildIntervalCard(_result!),
          ],
        ],
      ),
    );
  }

  // ─── Upload card ────────────────────────────────────────────────────────────

  Widget _buildUploadCard() {
    final uploaded = _uploaded;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.upload_file, color: AppColors.primary, size: 20),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'Previous Footfall & Orders Data',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
                ),
              ),
              if (_datasets.isNotEmpty)
                TextButton.icon(
                  onPressed: _loadDatasets,
                  icon: const Icon(Icons.refresh, size: 16),
                  label: const Text('Refresh'),
                ),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
            'Upload a CSV, TSV, TXT or Excel file of past footfall and orders. '
            'The more history you add, the more accurate the demand forecast.',
            style: TextStyle(fontSize: 12, color: AppColors.textMuted),
          ),
          const SizedBox(height: 14),

          if (_pendingFile != null) _buildPendingFile(),
          if (uploaded != null) _buildUploadSummary(uploaded),
          if (_pendingFile == null && uploaded == null) _buildFormatHint(),

          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _uploading ? null : _chooseFile,
                  icon: const Icon(Icons.attach_file, size: 18),
                  label: Text(_pendingFile == null ? 'Choose File' : 'Change File'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: (_pendingFile == null || _uploading) ? null : _upload,
                  icon: _uploading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Icon(Icons.cloud_upload_outlined, size: 18),
                  label: Text(_uploading ? 'Uploading...' : 'Upload'),
                ),
              ),
            ],
          ),

          if (_uploadError != null) ...[
            const SizedBox(height: 12),
            AuthErrorSurface(message: _uploadError!),
          ],

          if (_datasets.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text(
              'Already uploaded',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textSecondary),
            ),
            const SizedBox(height: 8),
            ..._datasets.take(3).map(_buildDatasetRow),
          ],
        ],
      ),
    );
  }

  Widget _buildFormatHint() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Expected columns (header names are matched loosely):',
            style: TextStyle(fontSize: 11, color: AppColors.textMuted),
          ),
          const SizedBox(height: 6),
          const Text(
            'date · attendance / headcount / footfall · orders · prepared_kg · consumed_kg · waste_kg',
            style: TextStyle(fontSize: 11, color: AppColors.textSecondary, fontFamily: 'monospace'),
          ),
        ],
      ),
    );
  }

  Widget _buildPendingFile() {
    final file = _pendingFile!;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.primarySurface,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const Icon(Icons.description_outlined, color: AppColors.primary, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              file.name,
              style: const TextStyle(color: AppColors.textPrimary, fontSize: 13),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close, size: 18, color: AppColors.textMuted),
            onPressed: _uploading ? null : () => setState(() => _pendingFile = null),
          ),
        ],
      ),
    );
  }

  Widget _buildUploadSummary(DatasetUploadResult r) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.good.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.good.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.check_circle_outline, color: AppColors.good, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '${r.rowsImported} rows imported',
                  style: const TextStyle(color: AppColors.textPrimary, fontSize: 14, fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          if (r.columnsFound.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: r.columnsFound
                  .map((c) => Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: AppColors.surfaceElevated,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Text(
                          c,
                          style: const TextStyle(fontSize: 10, color: AppColors.textSecondary),
                        ),
                      ))
                  .toList(),
            ),
          ],
          if (r.dateRangeDays > 0) ...[
            const SizedBox(height: 8),
            Text(
              'Covers ${r.earliestDate} to ${r.latestDate} (${r.dateRangeDays} days)',
              style: const TextStyle(fontSize: 11, color: AppColors.textSecondary),
            ),
          ],
          if (r.hasRejectedRows) ...[
            const SizedBox(height: 8),
            Text(
              '${r.rowsRejected} rows were skipped because they could not be read.',
              style: const TextStyle(fontSize: 11, color: AppColors.warning),
            ),
            ...r.rejectSamples.take(3).map(
                  (s) => Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      '• $s',
                      style: const TextStyle(fontSize: 10, color: AppColors.textMuted),
                    ),
                  ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildDatasetRow(DatasetInfo d) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          const Icon(Icons.table_rows_outlined, size: 16, color: AppColors.textMuted),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              d.filename,
              style: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Text(
            '${d.rowsImported} rows',
            style: const TextStyle(fontSize: 11, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }

  // ─── Manual input ───────────────────────────────────────────────────────────

  Widget _buildInputCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'For this meal',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
          ),
          const SizedBox(height: 4),
          const Text(
            'Optional. Leave attendance blank to predict from your uploaded history alone.',
            style: TextStyle(fontSize: 11, color: AppColors.textMuted),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _attendanceController,
            keyboardType: TextInputType.number,
            style: const TextStyle(color: AppColors.textPrimary),
            decoration: const InputDecoration(
              labelText: 'Expected attendance',
              prefixIcon: Icon(Icons.people_outline, color: AppColors.textMuted),
            ),
          ),
          const SizedBox(height: 16),
          const Text('Meal Type', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            children: MealType.values.map((meal) {
              return ChoiceChip(
                label: Text(meal.label),
                selected: meal == _selectedMeal,
                onSelected: (_) => setState(() => _selectedMeal = meal),
                selectedColor: AppColors.primarySurface,
                checkmarkColor: AppColors.primary,
              );
            }).toList(),
          ),
          const SizedBox(height: 16),
          const Text('Menu Items', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _menuOptions.map((item) {
              final selected = _selectedMenu.contains(item);
              return FilterChip(
                label: Text(item[0].toUpperCase() + item.substring(1)),
                selected: selected,
                onSelected: (val) => setState(() {
                  if (val) {
                    _selectedMenu.add(item);
                  } else {
                    _selectedMenu.remove(item);
                  }
                }),
                selectedColor: AppColors.primarySurface,
                checkmarkColor: AppColors.primary,
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  // ─── Result ────────────────────────────────────────────────────────────────

  Widget _buildResultCard(Prediction p) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: AppColors.primaryGradient,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: AppColors.primary.withOpacity(0.25), blurRadius: 20, offset: const Offset(0, 8))],
      ),
      child: Column(
        children: [
          const Text('How much food to prepare', style: TextStyle(fontSize: 14, color: Colors.white70)),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _resultMetric('Consumption', '${p.predictedConsumption.round()}', 'kg'),
              Container(width: 1, height: 40, color: Colors.white24),
              _resultMetric('Production', '${p.recommendedProduction.round()}', 'kg'),
              Container(width: 1, height: 40, color: Colors.white24),
              _resultMetric('Surplus', '${p.expectedSurplus.round()}', 'kg'),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              StatusBadge(label: p.surplusRisk, color: AppColors.statusColor(p.surplusRisk)),
              const SizedBox(width: 12),
              StatusBadge(label: p.dataSource, color: AppColors.info, icon: Icons.data_object),
            ],
          ),
        ],
      ),
    );
  }

  Widget _resultMetric(String label, String value, String unit) {
    return Column(
      children: [
        RichText(text: TextSpan(children: [
          TextSpan(text: value, style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w700, color: Colors.white)),
          TextSpan(text: ' $unit', style: const TextStyle(fontSize: 13, color: Colors.white70)),
        ])),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 12, color: Colors.white70)),
      ],
    );
  }

  Widget _buildIntervalCard(Prediction p) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Prediction Interval',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('p10: ${p.interval.p10.round()} kg', style: const TextStyle(fontSize: 12, color: AppColors.good)),
              Text('p50: ${p.interval.p50.round()} kg', style: const TextStyle(fontSize: 12, color: AppColors.textPrimary)),
              Text('p90: ${p.interval.p90.round()} kg', style: const TextStyle(fontSize: 12, color: AppColors.danger)),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Model: ${p.modelVersion} · coverage ${(p.interval.coverageTarget * 100).round()}%',
            style: const TextStyle(fontSize: 11, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }
}

/// Compact error surface, so this screen does not depend on the auth widgets.
class AuthErrorSurface extends StatelessWidget {
  final String message;
  const AuthErrorSurface({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.danger.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.danger.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: AppColors.danger, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(message, style: const TextStyle(color: AppColors.danger, fontSize: 13)),
          ),
        ],
      ),
    );
  }
}