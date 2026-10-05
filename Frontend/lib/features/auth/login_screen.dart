import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/api/api_client.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/auth_service.dart';
import 'auth_provider.dart';
import 'widgets/auth_scaffold.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _submit() {
    FocusScope.of(context).unfocus();
    if (!_formKey.currentState!.validate()) return;
    ref.read(authProvider.notifier).login(
          _emailController.text.trim(),
          _passwordController.text,
        );
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);

    return AuthScaffold(
      title: 'Annapurna',
      subtitle: 'Food Supply Chain Intelligence',
      footer: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            'New here?',
            style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
          ),
          TextButton(
            onPressed: authState.isLoading ? null : () => context.go('/signup'),
            child: const Text('Create Account'),
          ),
        ],
      ),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextFormField(
              controller: _emailController,
              enabled: !authState.isLoading,
              keyboardType: TextInputType.emailAddress,
              autocorrect: false,
              style: const TextStyle(color: AppColors.textPrimary),
              decoration: const InputDecoration(
                labelText: 'Email',
                prefixIcon: Icon(Icons.email_outlined, color: AppColors.textMuted),
              ),
              validator: (v) {
                final value = v?.trim() ?? '';
                if (value.isEmpty) return 'Please enter your email';
                if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(value)) {
                  return 'Enter a valid email address';
                }
                return null;
              },
            ),
            const SizedBox(height: 14),
            TextFormField(
              controller: _passwordController,
              enabled: !authState.isLoading,
              obscureText: _obscurePassword,
              style: const TextStyle(color: AppColors.textPrimary),
              decoration: InputDecoration(
                labelText: 'Password',
                prefixIcon: const Icon(Icons.lock_outline_rounded, color: AppColors.textMuted),
                suffixIcon: IconButton(
                  icon: Icon(
                    _obscurePassword ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                    color: AppColors.textMuted,
                  ),
                  onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                ),
              ),
              validator: (v) => (v == null || v.isEmpty) ? 'Please enter your password' : null,
              onFieldSubmitted: (_) => _submit(),
            ),
            const SizedBox(height: 24),
            AuthSubmitButton(
              label: 'Sign In',
              isLoading: authState.isLoading,
              onPressed: _submit,
            ),
            if (authState.error != null) ...[
              const SizedBox(height: 16),
              AuthErrorBanner(message: authState.error!),
            ],
            const SizedBox(height: 24),
            _DemoAccounts(
                isLoading: authState.isLoading,
                onDemoTap: (role) {
                  final account = demoAccounts[role]!;
                  _emailController.text = account.email;
                  _passwordController.text = account.password;
                  ref.read(authProvider.notifier).loginAsRole(role);
                },
              ),
          ],
        ),
      ),
    );
  }
}

class _DemoAccounts extends StatelessWidget {
  final bool isLoading;
  final ValueChanged<UserRole> onDemoTap;

  const _DemoAccounts({required this.isLoading, required this.onDemoTap});

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        title: const Text(
          'Demo Accounts',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: AppColors.textSecondary,
          ),
        ),
        subtitle: const Text(
          'Quick login for demonstration',
          style: TextStyle(fontSize: 12, color: AppColors.textMuted),
        ),
        iconColor: AppColors.textSecondary,
        collapsedIconColor: AppColors.textSecondary,
        collapsedShape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: AppColors.border.withOpacity(0.5)),
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: AppColors.border.withOpacity(0.5)),
        ),
        backgroundColor: AppColors.surface.withOpacity(0.6),
        collapsedBackgroundColor: AppColors.surface.withOpacity(0.6),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: [
          Row(
            children: [
              Expanded(child: _roleButton(UserRole.kitchen, Icons.restaurant, 'Kitchen', const Color(0xFF10B981))),
              const SizedBox(width: 10),
              Expanded(child: _roleButton(UserRole.ngo, Icons.volunteer_activism, 'NGO', const Color(0xFF3B82F6))),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(child: _roleButton(UserRole.logistics, Icons.local_shipping, 'Driver', const Color(0xFFF59E0B))),
              const SizedBox(width: 10),
              Expanded(child: _roleButton(UserRole.admin, Icons.admin_panel_settings, 'Admin', const Color(0xFF8B5CF6))),
            ],
          ),
        ],
      ),
    );
  }

  Widget _roleButton(UserRole role, IconData icon, String label, Color color) {
    return Material(
      color: color.withOpacity(0.1),
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: isLoading ? null : () => onDemoTap(role),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: color.withOpacity(0.3)),
          ),
          child: Column(
            children: [
              Icon(icon, color: color, size: 26),
              const SizedBox(height: 6),
              Text(label, style: TextStyle(color: color, fontSize: 13, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      ),
    );
  }
}