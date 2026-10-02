import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/waste_service.dart';

class WasteLogScreen extends ConsumerStatefulWidget {
  const WasteLogScreen({super.key});

  @override
  ConsumerState<WasteLogScreen> createState() => _WasteLogScreenState();
}

class _WasteLogScreenState extends ConsumerState<WasteLogScreen> {
  final _quantityController = TextEditingController();
  WasteCause _cause = WasteCause.overproduction;
  MealType _mealType = MealType.lunch;
  bool _loading = false;
  Map<String, dynamic>? _analytics;
  bool _analyticsLoading = true;

  @override
  void initState() {
    super.initState();
    _loadAnalytics();
  }

  Future<void> _loadAnalytics() async {
    final service = ref.read(wasteServiceProvider);
    final result = await service.getAnalytics();
    result.when(
      success: (data) => setState(() { _analytics = data; _analyticsLoading = false; }),
      failure: (_) => setState(() => _analyticsLoading = false),
    );
  }

  Future<void> _submit() async {
    if (_quantityController.text.isEmpty) return;
    setState(() => _loading = true);
    final service = ref.read(wasteServiceProvider);
    final result = await service.logWaste(
      quantityKg: double.tryParse(_quantityController.text) ?? 0,
      cause: _cause.apiValue,
      mealType: _mealType.apiValue,
    );
    result.when(
      success: (_) {
        _quantityController.clear();
        setState(() => _loading = false);
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Waste logged')));
        _loadAnalytics();
      },
      failure: (e) {
        setState(() => _loading = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      },
    );
  }

  @override
  void dispose() {
    _quantityController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Waste Log')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.circular(16), border: Border.all(color: AppColors.border, width: 0.5)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Log Waste', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                const SizedBox(height: 16),
                TextField(
                  controller: _quantityController,
                  keyboardType: TextInputType.number,
                  style: const TextStyle(color: AppColors.textPrimary),
                  decoration: const InputDecoration(labelText: 'Quantity (kg)', prefixIcon: Icon(Icons.scale, color: AppColors.textMuted)),
                ),
                const SizedBox(height: 16),
                const Text('Cause', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: WasteCause.values.map((c) => ChoiceChip(
                    label: Text(c.label, style: const TextStyle(fontSize: 12)),
                    selected: _cause == c,
                    onSelected: (_) => setState(() => _cause = c),
                    selectedColor: AppColors.primarySurface,
                  )).toList(),
                ),
                const SizedBox(height: 16),
                const Text('Meal', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: MealType.values.map((m) => ChoiceChip(
                    label: Text(m.label),
                    selected: _mealType == m,
                    onSelected: (_) => setState(() => _mealType = m),
                    selectedColor: AppColors.primarySurface,
                  )).toList(),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton.icon(
              onPressed: _loading ? null : _submit,
              icon: _loading
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.save),
              label: Text(_loading ? 'Logging...' : 'Log Waste'),
            ),
          ),
          if (_analytics != null) ...[
            const SizedBox(height: 24),
            _buildAnalyticsCard(_analytics!),
          ],
        ],
      ),
    );
  }

  Widget _buildAnalyticsCard(Map<String, dynamic> data) {
    final byCause = (data['by_cause'] as Map<String, dynamic>?) ?? {};
    final totalKg = data['total_kg'] ?? 0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.circular(16), border: Border.all(color: AppColors.border, width: 0.5)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Waste Analytics', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
          const SizedBox(height: 16),
          Text('Total: ${totalKg} kg', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w700, color: AppColors.danger)),
          const SizedBox(height: 16),
          ...byCause.entries.map((entry) {
            final pct = totalKg > 0 ? (entry.value as num).toDouble() / (totalKg as num).toDouble() : 0.0;
            return Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(entry.key.replaceAll('_', ' '), style: const TextStyle(fontSize: 13, color: AppColors.textSecondary)),
                      Text('${entry.value} kg', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                    ],
                  ),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      value: pct,
                      backgroundColor: AppColors.surfaceElevated,
                      valueColor: AlwaysStoppedAnimation<Color>(AppColors.danger.withValues(alpha: 0.7)),
                      minHeight: 6,
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}
