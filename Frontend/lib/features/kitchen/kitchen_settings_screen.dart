import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/theme/app_colors.dart';
import '../auth/auth_provider.dart';
import '../../data/dtos/models.dart';

/// Kitchen-only settings.
///
/// Kept deliberately small: this app does not offer profile editing, password
/// change, notification topics, theme switching, or unit preferences, so the
/// page only exposes what the kitchen role can actually use — account info,
/// sign out, the push-notification permission the app already requests, and
/// the app version that the existing More → About snackbar already references.
class KitchenSettingsScreen extends ConsumerWidget {
  const KitchenSettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(currentUserProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _accountSection(context, ref, user),
          const SizedBox(height: 24),
          _notificationsSection(context, ref),
          const SizedBox(height: 24),
          _dataSection(context, ref),
          const SizedBox(height: 24),
          _aboutSection(context),
          const SizedBox(height: 16),
          _signOut(context, ref),
        ],
      ),
    );
  }

  Widget _accountSection(BuildContext context, WidgetRef ref, User? user) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('Account'),
        const SizedBox(height: 12),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border, width: 0.5),
          ),
          child: Column(
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(Icons.person, color: Colors.white, size: 26),
              ),
              const SizedBox(height: 14),
              Text(
                user?.displayName ?? 'Kitchen',
                style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
              ),
              const SizedBox(height: 4),
              Text(
                user?.email ?? '',
                style: const TextStyle(fontSize: 13, color: AppColors.textSecondary),
              ),
              const SizedBox(height: 4),
              Text(
                (user?.role.name ?? 'KITCHEN').toUpperCase(),
                style: const TextStyle(fontSize: 11, color: AppColors.textMuted, letterSpacing: 1),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _notificationsSection(BuildContext context, WidgetRef ref) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('Notifications'),
        const SizedBox(height: 12),
        _settingsTile(
          context,
          icon: Icons.notifications_outlined,
          title: 'Push notifications',
          subtitle: 'Alerts when a surplus needs quality approval',
          onTap: () => _showPermissionInfo(context),
        ),
      ],
    );
  }

  Widget _dataSection(BuildContext context, WidgetRef ref) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('Data & sync'),
        const SizedBox(height: 12),
        _settingsTile(
          context,
          icon: Icons.cloud_sync_outlined,
          title: 'Offline sync',
          subtitle: 'Pending changes are sent when the app is back online',
          onTap: () {},
        ),
      ],
    );
  }

  Widget _aboutSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('About'),
        const SizedBox(height: 12),
        _settingsTile(
          context,
          icon: Icons.info_outline,
          title: 'Annapurna Kitchen Portal',
          subtitle: 'Version 1.0',
          onTap: () {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Annapurna Kitchen Portal v1.0')),
            );
          },
        ),
      ],
    );
  }

  Widget _signOut(BuildContext context, WidgetRef ref) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      child: SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.danger,
            side: const BorderSide(color: AppColors.danger),
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
          onPressed: () async {
            final confirmed = await _confirm(context, 'Sign out?', 'You will need to sign in again to continue.');
            if (!confirmed) return;
            await ref.read(authProvider.notifier).logout();
          },
          icon: const Icon(Icons.logout, size: 20),
          label: const Text('Sign Out'),
        ),
      ),
    );
  }

  Future<void> _showPermissionInfo(BuildContext context) async {
    await showDialog(
      context: context,
      builder: (dialog) => AlertDialog(
        backgroundColor: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Push notifications', style: TextStyle(color: AppColors.textPrimary)),
        content: const Text(
          'The app asks for notification permission so you can be alerted when action is needed — for example, when a surplus is waiting for a quality decision.',
          style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialog).pop(),
            child: const Text('OK', style: TextStyle(color: AppColors.primary)),
          ),
        ],
      ),
    );
  }

  Future<bool> _confirm(BuildContext context, String title, String message) async {
    final result = await showDialog<bool>(
      context: context,
      builder: (dialog) => AlertDialog(
        backgroundColor: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text(title, style: const TextStyle(color: AppColors.textPrimary)),
        content: Text(message, style: const TextStyle(color: AppColors.textSecondary, fontSize: 14)),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialog).pop(false),
            child: const Text('Cancel', style: TextStyle(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialog).pop(true),
            style: TextButton.styleFrom(foregroundColor: AppColors.danger),
            child: const Text('Sign out'),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  Widget _sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Text(
        title,
        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textMuted, letterSpacing: 0.5),
      ),
    );
  }

  Widget _settingsTile(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 2),
      child: ListTile(
        leading: Icon(icon, color: AppColors.textSecondary, size: 22),
        title: Text(title, style: const TextStyle(color: AppColors.textPrimary, fontSize: 15, fontWeight: FontWeight.w500)),
        subtitle: Text(subtitle, style: const TextStyle(color: AppColors.textMuted, fontSize: 12)),
        trailing: const Icon(Icons.chevron_right, color: AppColors.textMuted, size: 20),
        onTap: onTap,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}
