import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/theme/app_colors.dart';
import '../../data/dtos/models.dart';
import '../../data/services/kitchen_service.dart';
import '../../shared/widgets/stat_card.dart';

class ImpactScreen extends ConsumerStatefulWidget {
  const ImpactScreen({super.key});

  @override
  ConsumerState<ImpactScreen> createState() => _ImpactScreenState();
}

class _ImpactScreenState extends ConsumerState<ImpactScreen> {
  ImpactData? _data;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final service = ref.read(kitchenServiceProvider);
    final result = await service.getImpact();
    result.when(
      success: (data) => setState(() { _data = data; _loading = false; }),
      failure: (_) => setState(() => _loading = false),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Impact Dashboard')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _data == null
              ? const Center(child: Text('Failed to load impact data', style: TextStyle(color: AppColors.textSecondary)))
              : _buildContent(_data!),
    );
  }

  Widget _buildContent(ImpactData d) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            gradient: AppColors.primaryGradient,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: AppColors.primary.withValues(alpha: 0.3), blurRadius: 24, offset: const Offset(0, 8))],
          ),
          child: Column(
            children: [
              const Icon(Icons.eco, size: 48, color: Colors.white),
              const SizedBox(height: 12),
              Text('${d.estimatedCarbonSavedKg.round()}', style: const TextStyle(fontSize: 48, fontWeight: FontWeight.w800, color: Colors.white, height: 1)),
              const SizedBox(height: 4),
              const Text('kg CO₂ saved', style: TextStyle(fontSize: 16, color: Colors.white70)),
              const SizedBox(height: 8),
              Text('≈ ${d.mealsEquivalent} meals equivalent', style: const TextStyle(fontSize: 14, color: Colors.white54)),
            ],
          ),
        ),
        const SizedBox(height: 16),
        GridView.count(
          crossAxisCount: 2,
          mainAxisSpacing: 12,
          crossAxisSpacing: 12,
          childAspectRatio: 1.5,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            StatCard(title: 'Waste Avoided', value: '${d.wasteAvoidedKg.round()}', unit: 'kg', icon: Icons.trending_down, color: AppColors.good),
            StatCard(title: 'Redistributed', value: '${d.redistributedKg.round()}', unit: 'kg', icon: Icons.volunteer_activism, color: AppColors.info),
            StatCard(title: 'Diverted', value: '${d.divertedKg.round()}', unit: 'kg', icon: Icons.recycling, color: AppColors.accent),
            StatCard(title: 'Redistributions', value: '${d.successfulRedistributions}', icon: Icons.check_circle_outline, color: AppColors.primary),
          ],
        ),
        const SizedBox(height: 20),
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.circular(16), border: Border.all(color: AppColors.border, width: 0.5)),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('14-Day Trend', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
              const SizedBox(height: 20),
              SizedBox(
                height: 150,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: d.trend.map((t) {
                    final maxVal = d.trend.map((x) => x.value).reduce((a, b) => a > b ? a : b);
                    final heightFraction = maxVal > 0 ? t.value / maxVal : 0.0;
                    return Expanded(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 2),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            Text('${t.value.round()}', style: const TextStyle(fontSize: 8, color: AppColors.textMuted)),
                            const SizedBox(height: 4),
                            Container(
                              height: 120 * heightFraction,
                              decoration: BoxDecoration(
                                gradient: AppColors.primaryGradient,
                                borderRadius: BorderRadius.circular(4),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Text(
          'Carbon factor: ${d.carbonFactorUsed} kg CO₂/kg food (${d.carbonFactorSource})',
          style: const TextStyle(fontSize: 11, color: AppColors.textMuted),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}
