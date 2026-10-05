import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/auth_service.dart';
import 'auth_provider.dart';
import 'widgets/auth_scaffold.dart';

/// Create-account screen.
///
/// The role is chosen here rather than inferred, because it decides both the
/// dashboard the router sends the user to and which organisation scope the
/// backend provisions for the account.
class SignupScreen extends ConsumerStatefulWidget {
  const SignupScreen({super.key});

  @override
  ConsumerState<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends ConsumerState<SignupScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmController = TextEditingController();
  final _organisationController = TextEditingController();

  UserRole _role = UserRole.kitchen;
  bool _obscurePassword = true;

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmController.dispose();
    _organisationController.dispose();
    super.dispose();
  }

  bool get _needsOrganisation =>
      _role == UserRole.kitchen || _role == UserRole.ngo;

  void _submit() {
    FocusScope.of(context).unfocus();
    if (!_formKey.currentState!.validate()) return;
    ref.read(authProvider.notifier).register(
          SignupRequest(
            name: _nameController.text.trim(),
            email: _emailController.text.trim(),
            password: _passwordController.text,
            role: _role,
            organisationName: _organisationController.text.trim(),
          ),
        );
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);

    return AuthScaffold(
      title: 'Create account',
      subtitle: 'Register to join the Annapurna food redistribution network',
      footer: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            'Already registered?',
            style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
          ),
          TextButton(
            onPressed: authState.isLoading ? null : () => context.go('/login'),
            child: const Text('Sign In'),
          ),
        ],
      ),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _FieldLabel('I am signing up as'),
            const SizedBox(height: 10),
            _RolePicker(
              selected: _role,
              enabled: !authState.isLoading,
              onChanged: (role) => setState(() => _role = role),
            ),
            const SizedBox(height: 24),
            TextFormField(
              controller: _nameController,
              enabled: !authState.isLoading,
              textCapitalization: TextCapitalization.words,
              style: const TextStyle(color: AppColors.textPrimary),
              decoration: const InputDecoration(
                labelText: 'Full name',
                prefixIcon: Icon(Icons.person_outline, color: AppColors.textMuted),
              ),
              validator: (v) =>
                  (v == null || v.trim().isEmpty) ? 'Please enter your name' : null,
            ),
            const SizedBox(height: 14),
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
              validator: _validatePassword,
            ),
            const SizedBox(height: 14),
            TextFormField(
              controller: _confirmController,
              enabled: !authState.isLoading,
              obscureText: _obscurePassword,
              style: const TextStyle(color: AppColors.textPrimary),
              decoration: const InputDecoration(
                labelText: 'Confirm password',
                prefixIcon: Icon(Icons.lock_reset_rounded, color: AppColors.textMuted),
              ),
              validator: (v) {
                if (v == null || v.isEmpty) return 'Please confirm your password';
                if (v != _passwordController.text) return 'Passwords do not match';
                return null;
              },
            ),
            // The backend creates the organisation (and the kitchen or recipient
            // row) for kitchen and NGO accounts, so this field only names it.
            if (_needsOrganisation) ...[
              const SizedBox(height: 14),
              TextFormField(
                controller: _organisationController,
                enabled: !authState.isLoading,
                textCapitalization: TextCapitalization.words,
                style: const TextStyle(color: AppColors.textPrimary),
                decoration: InputDecoration(
                  labelText: _role == UserRole.kitchen
                      ? 'Kitchen name (optional)'
                      : 'Organisation name (optional)',
                  prefixIcon: const Icon(Icons.apartment_outlined, color: AppColors.textMuted),
                  helperText: 'Leave blank to use your name',
                  helperStyle: const TextStyle(color: AppColors.textMuted, fontSize: 12),
                ),
              ),
            ],
            const SizedBox(height: 24),
            AuthSubmitButton(
              label: 'Create Account',
              isLoading: authState.isLoading,
              onPressed: _submit,
            ),
            if (authState.error != null) ...[
              const SizedBox(height: 16),
              AuthErrorBanner(message: authState.error!),
            ],
            const SizedBox(height: 12),
            const Text(
              'Your account is signed in as soon as it is created.',
              textAlign: TextAlign.center,
              style: TextStyle(color: AppColors.textMuted, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }

  String? _validatePassword(String? value) {
    final password = value ?? '';
    if (password.isEmpty) return 'Please choose a password';
    if (password.length < 8) return 'Use at least 8 characters';
    final hasLetter = password.contains(RegExp(r'[A-Za-z]'));
    final hasOther = password.contains(RegExp(r'[^A-Za-z]'));
    if (!hasLetter || !hasOther) {
      return 'Mix letters with digits or symbols';
    }
    return null;
  }
}

class _FieldLabel extends StatelessWidget {
  final String text;
  const _FieldLabel(this.text);

  @override
  Widget build(BuildContext context) => Text(
        text,
        style: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: AppColors.textSecondary,
        ),
      );
}

/// Role selector. ADMIN is offered because this deployment allows self-service
/// admin accounts; the backend rejects any role outside the allow-list.
class _RolePicker extends StatelessWidget {
  final UserRole selected;
  final bool enabled;
  final ValueChanged<UserRole> onChanged;

  const _RolePicker({
    required this.selected,
    required this.enabled,
    required this.onChanged,
  });

  static const _roles = [
    (UserRole.kitchen, Icons.restaurant, 'Kitchen', Color(0xFF10B981)),
    (UserRole.ngo, Icons.volunteer_activism, 'NGO', Color(0xFF3B82F6)),
    (UserRole.logistics, Icons.local_shipping, 'Driver', Color(0xFFF59E0B)),
    (UserRole.admin, Icons.admin_panel_settings, 'Admin', Color(0xFF8B5CF6)),
  ];

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(child: _tile(_roles[0])),
            const SizedBox(width: 10),
            Expanded(child: _tile(_roles[1])),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(child: _tile(_roles[2])),
            const SizedBox(width: 10),
            Expanded(child: _tile(_roles[3])),
          ],
        ),
      ],
    );
  }

  Widget _tile((UserRole, IconData, String, Color) role) {
    final isSelected = selected == role.$1;
    final color = role.$4;
    return Material(
      color: color.withOpacity(isSelected ? 0.22 : 0.08),
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: enabled ? () => onChanged(role.$1) : null,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: isSelected ? color : color.withOpacity(0.3),
              width: isSelected ? 1.5 : 1,
            ),
          ),
          child: Column(
            children: [
              Icon(role.$2, color: color, size: 26),
              const SizedBox(height: 6),
              Text(
                role.$3,
                style: TextStyle(
                  color: color,
                  fontSize: 13,
                  fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}