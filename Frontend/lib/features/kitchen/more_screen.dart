import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';
import '../auth/auth_provider.dart';

class KitchenMoreScreen extends ConsumerWidget {
  const KitchenMoreScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(currentUserProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('More')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: AppColors.primaryGradient,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: const Icon(Icons.person, color: Colors.white, size: 28),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(user?.displayName ?? '', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: Colors.white)),
                      Text(user?.email ?? '', style: const TextStyle(fontSize: 13, color: Colors.white70)),
                      Text(user?.role.name.toUpperCase() ?? '', style: const TextStyle(fontSize: 11, color: Colors.white54, letterSpacing: 1)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          _sectionTitle('Kitchen Tools'),
          _menuItem(context, Icons.eco, 'Impact Dashboard', () => context.push('/kitchen/impact')),
          _menuItem(context, Icons.sensors, 'Sensors', () => context.push('/kitchen/sensors')),
          _menuItem(context, Icons.delete_sweep_outlined, 'Waste Log', () => context.push('/kitchen/waste')),
          _menuItem(context, Icons.tune, 'What-If Analysis', () => context.push('/kitchen/what-if')),
          const SizedBox(height: 24),
          _sectionTitle('Account'),
          _menuItem(context, Icons.notifications_outlined, 'Alerts', () => context.push('/kitchen/alerts')),
          _menuItem(context, Icons.settings_outlined, 'Settings', () => context.push('/kitchen/settings')),
          _menuItem(context, Icons.info_outline, 'About', () => context.push('/kitchen/settings')),
          const SizedBox(height: 16),
          _menuItem(
            context,
            Icons.logout,
            'Sign Out',
            () async {
              await ref.read(authProvider.notifier).logout();
            },
            color: AppColors.danger,
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textMuted, letterSpacing: 0.5)),
    );
  }

  Widget _menuItem(BuildContext context, IconData icon, String label, VoidCallback onTap, {Color? color}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 2),
      child: ListTile(
        leading: Icon(icon, color: color ?? AppColors.textSecondary, size: 22),
        title: Text(label, style: TextStyle(color: color ?? AppColors.textPrimary, fontSize: 15)),
        trailing: Icon(Icons.chevron_right, color: color?.withOpacity(0.5) ?? AppColors.textMuted, size: 20),
        onTap: onTap,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}
