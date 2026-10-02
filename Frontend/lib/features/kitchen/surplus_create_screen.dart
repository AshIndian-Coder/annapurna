import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/surplus_service.dart';

class SurplusCreateScreen extends ConsumerStatefulWidget {
  const SurplusCreateScreen({super.key});

  @override
  ConsumerState<SurplusCreateScreen> createState() => _SurplusCreateScreenState();
}

class _SurplusCreateScreenState extends ConsumerState<SurplusCreateScreen> {
  final _nameController = TextEditingController();
  final _quantityController = TextEditingController();
  MealType _mealType = MealType.lunch;
  String _category = 'Grains';
  DateTime _expiryDate = DateTime.now().add(const Duration(hours: 8));
  bool _loading = false;

  final _categories = ['Grains', 'Lentils', 'Dairy', 'Vegetables', 'Fruits', 'Prepared', 'Beverages'];

  @override
  void dispose() {
    _nameController.dispose();
    _quantityController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (_nameController.text.isEmpty || _quantityController.text.isEmpty) return;
    setState(() => _loading = true);
    final service = ref.read(surplusServiceProvider);
    final result = await service.createSurplus(
      foodName: _nameController.text,
      quantityKg: double.tryParse(_quantityController.text) ?? 0,
      preparedAt: DateTime.now().toIso8601String(),
      expiryAt: _expiryDate.toIso8601String(),
      foodCategory: _category,
    );
    result.when(
      success: (_) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Surplus created successfully')),
        );
        context.pop();
      },
      failure: (error) {
        setState(() => _loading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(error.message), backgroundColor: AppColors.danger),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Create Surplus')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border, width: 0.5),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _nameController,
                  style: const TextStyle(color: AppColors.textPrimary),
                  decoration: const InputDecoration(labelText: 'Food Name', prefixIcon: Icon(Icons.restaurant, color: AppColors.textMuted)),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _quantityController,
                  keyboardType: TextInputType.number,
                  style: const TextStyle(color: AppColors.textPrimary),
                  decoration: const InputDecoration(labelText: 'Quantity (kg)', prefixIcon: Icon(Icons.scale, color: AppColors.textMuted)),
                ),
                const SizedBox(height: 16),
                const Text('Category', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _categories.map((c) => ChoiceChip(
                    label: Text(c),
                    selected: _category == c,
                    onSelected: (_) => setState(() => _category = c),
                    selectedColor: AppColors.primarySurface,
                    checkmarkColor: AppColors.primary,
                  )).toList(),
                ),
                const SizedBox(height: 16),
                const Text('Meal Type', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: MealType.values.map((m) => ChoiceChip(
                    label: Text(m.label),
                    selected: _mealType == m,
                    onSelected: (_) => setState(() => _mealType = m),
                    selectedColor: AppColors.primarySurface,
                    checkmarkColor: AppColors.primary,
                  )).toList(),
                ),
                const SizedBox(height: 16),
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: const Icon(Icons.schedule, color: AppColors.textMuted),
                  title: const Text('Expiry Time', style: TextStyle(color: AppColors.textSecondary, fontSize: 13)),
                  subtitle: Text(
                    '${_expiryDate.hour.toString().padLeft(2, '0')}:${_expiryDate.minute.toString().padLeft(2, '0')} — ${_expiryDate.difference(DateTime.now()).inHours}h from now',
                    style: const TextStyle(color: AppColors.textPrimary, fontSize: 15),
                  ),
                  trailing: const Icon(Icons.edit, size: 18, color: AppColors.textMuted),
                  onTap: () async {
                    final time = await showTimePicker(context: context, initialTime: TimeOfDay.fromDateTime(_expiryDate));
                    if (time != null) {
                      setState(() {
                        _expiryDate = DateTime(DateTime.now().year, DateTime.now().month, DateTime.now().day, time.hour, time.minute);
                        if (_expiryDate.isBefore(DateTime.now())) {
                          _expiryDate = _expiryDate.add(const Duration(days: 1));
                        }
                      });
                    }
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton.icon(
              onPressed: _loading ? null : _submit,
              icon: _loading
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.add_circle_outline),
              label: Text(_loading ? 'Creating...' : 'Create Surplus'),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'The batch will enter PENDING_SAFETY status. A food safety check is required before redistribution.',
            style: TextStyle(fontSize: 12, color: AppColors.textMuted),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
