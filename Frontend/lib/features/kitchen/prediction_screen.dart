import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../data/dtos/models.dart';
import '../../data/services/kitchen_service.dart';
import '../../shared/widgets/common_widgets.dart';

class PredictionScreen extends ConsumerStatefulWidget {
  const PredictionScreen({super.key});

  @override
  ConsumerState<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends ConsumerState<PredictionScreen> {
  final _attendanceController = TextEditingController(text: '450');
  MealType _selectedMeal = MealType.lunch;
  final List<String> _selectedMenu = ['rice', 'dal', 'paneer'];
  Prediction? _result;
  bool _loading = false;

  final _menuOptions = ['rice', 'dal', 'paneer', 'roti', 'sabji', 'biryani', 'curd', 'salad', 'soup', 'sweet'];

  @override
  void dispose() {
    _attendanceController.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    setState(() => _loading = true);
    final service = ref.read(kitchenServiceProvider);
    final result = await service.predictDemand(
      attendance: int.tryParse(_attendanceController.text) ?? 450,
      mealType: _selectedMeal.apiValue,
      menu: _selectedMenu,
      dayOfWeek: DateTime.now().weekday - 1,
      date: DateTime.now().toIso8601String().substring(0, 10),
    );
    result.when(
      success: (data) => setState(() { _result = data; _loading = false; }),
      failure: (error) => setState(() { _loading = false; }),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Demand Prediction')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildInputCard(),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton.icon(
              onPressed: _loading ? null : _predict,
              icon: _loading
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.auto_graph),
              label: Text(_loading ? 'Predicting...' : 'Predict Demand'),
            ),
          ),
          if (_result != null) ...[
            const SizedBox(height: 24),
            _buildResultCard(_result!),
            const SizedBox(height: 16),
            _buildIntervalCard(_result!),
            const SizedBox(height: 16),
            _buildDriversCard(_result!),
          ],
        ],
      ),
    );
  }

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
          const Text('Input Parameters', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
          const SizedBox(height: 16),
          TextField(
            controller: _attendanceController,
            keyboardType: TextInputType.number,
            style: const TextStyle(color: AppColors.textPrimary),
            decoration: const InputDecoration(
              labelText: 'Expected Attendance',
              prefixIcon: Icon(Icons.people_outline, color: AppColors.textMuted),
            ),
          ),
          const SizedBox(height: 16),
          const Text('Meal Type', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            children: MealType.values.map((meal) {
              final selected = meal == _selectedMeal;
              return ChoiceChip(
                label: Text(meal.label),
                selected: selected,
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
                onSelected: (val) {
                  setState(() {
                    if (val) { _selectedMenu.add(item); } else { _selectedMenu.remove(item); }
                  });
                },
                selectedColor: AppColors.primarySurface,
                checkmarkColor: AppColors.primary,
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard(Prediction p) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: AppColors.primaryGradient,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: AppColors.primary.withValues(alpha: 0.25), blurRadius: 20, offset: const Offset(0, 8))],
      ),
      child: Column(
        children: [
          const Text('Prediction Result', style: TextStyle(fontSize: 14, color: Colors.white70)),
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
              StatusBadge(label: p.dataSource, color: p.dataSource == 'SYNTHETIC' ? AppColors.accent : AppColors.info, icon: Icons.data_object),
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
    final range = p.interval.p90 - p.interval.p10;
    final p50Pos = (p.interval.p50 - p.interval.p10) / range;

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
          const Text('Prediction Interval', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
          const SizedBox(height: 20),
          SizedBox(
            height: 60,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Positioned(
                  left: 0,
                  right: 0,
                  child: Container(
                    height: 8,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(4),
                      gradient: const LinearGradient(colors: [Color(0xFF10B981), Color(0xFFF59E0B), Color(0xFFEF4444)]),
                    ),
                  ),
                ),
                Positioned(
                  left: p50Pos * (MediaQuery.of(context).size.width - 72) - 1,
                  child: Column(
                    children: [
                      Container(width: 3, height: 24, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(2))),
                      const SizedBox(height: 4),
                      Text('p50: ${p.interval.p50.round()}', style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('p10: ${p.interval.p10.round()} kg', style: const TextStyle(fontSize: 12, color: AppColors.good)),
              Text('p90: ${p.interval.p90.round()} kg', style: const TextStyle(fontSize: 12, color: AppColors.danger)),
            ],
          ),
          const SizedBox(height: 8),
          Text('Coverage: ${(p.interval.coverageTarget * 100).round()}%', style: const TextStyle(fontSize: 12, color: AppColors.textMuted)),
        ],
      ),
    );
  }

  Widget _buildDriversCard(Prediction p) {
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
          const Text('Top Drivers', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
          const SizedBox(height: 16),
          ...p.topDrivers.map((d) => Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    d.feature[0].toUpperCase() + d.feature.substring(1).replaceAll('_', ' '),
                    style: const TextStyle(fontSize: 14, color: AppColors.textPrimary),
                  ),
                ),
                Text(
                  '${d.effectKg > 0 ? '+' : ''}${d.effectKg.toStringAsFixed(1)} kg',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: d.effectKg > 0 ? AppColors.danger : AppColors.good,
                  ),
                ),
              ],
            ),
          )),
          const SizedBox(height: 8),
          Text('Model: ${p.modelVersion}', style: const TextStyle(fontSize: 11, color: AppColors.textMuted)),
        ],
      ),
    );
  }
}
