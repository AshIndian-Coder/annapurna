import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/theme/app_colors.dart';
import '../../data/dtos/models.dart';
import '../../data/services/kitchen_service.dart';

class SensorsScreen extends ConsumerStatefulWidget {
  const SensorsScreen({super.key});

  @override
  ConsumerState<SensorsScreen> createState() => _SensorsScreenState();
}

class _SensorsScreenState extends ConsumerState<SensorsScreen> {
  List<SensorReading>? _readings;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final service = ref.read(kitchenServiceProvider);
    final result = await service.getSensorReadings();
    result.when(
      success: (data) => setState(() { _readings = data; _loading = false; }),
      failure: (_) => setState(() => _loading = false),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sensors')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _readings == null
              ? const Center(child: Text('No sensor data'))
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _readings!.length,
                  itemBuilder: (context, i) => _buildSensorCard(_readings![i]),
                ),
    );
  }

  Widget _buildSensorCard(SensorReading r) {
    final isDanger = r.isTempDangerZone;
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: isDanger ? AppColors.danger.withValues(alpha: 0.4) : AppColors.border, width: isDanger ? 1.5 : 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: (isDanger ? AppColors.danger : AppColors.info).withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(Icons.sensors, size: 20, color: isDanger ? AppColors.danger : AppColors.info),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(r.locationId.replaceAll('-', ' ').replaceAll('_', ' '), style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                    Text(r.sensorId, style: const TextStyle(fontSize: 11, color: AppColors.textMuted, fontFamily: 'monospace')),
                  ],
                ),
              ),
              if (isDanger)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(color: AppColors.danger.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(8)),
                  child: const Text('DANGER ZONE', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.danger)),
                ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _metric(Icons.thermostat, '${r.temperatureC}°C', isDanger ? AppColors.danger : AppColors.good),
              const SizedBox(width: 24),
              _metric(Icons.water_drop_outlined, '${r.humidityPct}%', AppColors.info),
              const SizedBox(width: 24),
              _metric(Icons.bolt, '${r.energyKwh} kWh', AppColors.accent),
            ],
          ),
        ],
      ),
    );
  }

  Widget _metric(IconData icon, String value, Color color) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 6),
        Text(value, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: color)),
      ],
    );
  }
}
