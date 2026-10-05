import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../features/auth/auth_provider.dart';
import '../../core/enums.dart';
import '../../features/auth/login_screen.dart';
import '../../features/auth/signup_screen.dart';
import '../../features/kitchen/kitchen_shell.dart';
import '../../features/kitchen/dashboard_screen.dart';
import '../../features/kitchen/prediction_screen.dart';
import '../../features/kitchen/surplus_list_screen.dart';
import '../../features/kitchen/surplus_create_screen.dart';
import '../../features/kitchen/quality_result_screen.dart';
import '../../features/kitchen/waste_log_screen.dart';
import '../../features/kitchen/impact_screen.dart';
import '../../features/kitchen/sensors_screen.dart';
import '../../features/kitchen/more_screen.dart';
import '../../features/kitchen/kitchen_settings_screen.dart';
import '../../features/ngo/ngo_shell.dart';
import '../../features/ngo/ngo_offers_screen.dart';
import '../../features/ngo/ngo_history_screen.dart';
import '../../features/ngo/ngo_scan_screen.dart';
import '../../features/logistics/logistics_shell.dart';
import '../../features/logistics/logistics_routes_screen.dart';
import '../../features/logistics/logistics_scan_screen.dart';
import '../../features/admin/admin_shell.dart';
import '../../features/admin/admin_dashboard_screen.dart';
import '../../features/admin/admin_models_screen.dart';
import '../../features/admin/admin_users_screen.dart';
import '../../features/admin/admin_audit_screen.dart';

/// Routes a freshly authenticated user to the dashboard for their role.
String? homeForRole(UserRole? role) => switch (role) {
      UserRole.kitchen => '/kitchen',
      UserRole.ngo => '/ngo',
      UserRole.logistics => '/logistics',
      UserRole.admin => '/admin',
      // A KITCHEN account is the only role that can be safely assumed; anything
      // else (including SYSTEM) would land on a screen it cannot use.
      _ => '/kitchen',
    };

final routerProvider = Provider<GoRouter>((ref) {
  final authState = ref.watch(authProvider);

  return GoRouter(
    initialLocation: '/login',
    redirect: (context, state) {
      final location = state.uri.toString();
      final isAuthRoute = location == '/login' || location == '/signup';

      // While the stored session is being restored, stay put. Redirecting here
      // would bounce a signed-in user to /login on every cold start.
      if (authState.isRestoring) return null;

      if (!authState.isAuthenticated) return isAuthRoute ? null : '/login';
      if (isAuthRoute) return homeForRole(authState.user?.role);
      return null;
    },
    routes: [
      GoRoute(path: '/login', builder: (_, __) => const LoginScreen()),
      GoRoute(path: '/signup', builder: (_, __) => const SignupScreen()),
      
      // Kitchen Role Routes
      ShellRoute(
        builder: (_, __, child) => KitchenShell(child: child),
        routes: [
          GoRoute(path: '/kitchen', builder: (_, __) => const DashboardScreen()),
          GoRoute(path: '/kitchen/predict', builder: (_, __) => const PredictionScreen()),
          GoRoute(path: '/kitchen/surplus', builder: (_, __) => const SurplusListScreen()),
          GoRoute(path: '/kitchen/more', builder: (_, __) => const KitchenMoreScreen()),
          GoRoute(path: '/kitchen/settings', builder: (_, __) => const KitchenSettingsScreen()),
        ],
      ),
      GoRoute(path: '/kitchen/surplus/create', builder: (_, __) => const SurplusCreateScreen()),
      GoRoute(
        path: '/kitchen/quality/result/:batchId',
        builder: (_, state) => QualityResultScreen(batchId: state.pathParameters['batchId']!),
      ),
      GoRoute(path: '/kitchen/waste', builder: (_, __) => const WasteLogScreen()),
      GoRoute(path: '/kitchen/impact', builder: (_, __) => const ImpactScreen()),
      GoRoute(path: '/kitchen/sensors', builder: (_, __) => const SensorsScreen()),
      GoRoute(path: '/kitchen/what-if', builder: (_, __) => const PredictionScreen()),
      GoRoute(path: '/kitchen/processing', builder: (_, __) => const ImpactScreen()),
      GoRoute(path: '/kitchen/alerts', builder: (_, __) => const SensorsScreen()),
      GoRoute(path: '/kitchen/qr-scan', builder: (_, __) => const Scaffold(body: Center(child: Text('QR Scanner')))),
      
      // NGO Role Routes
      ShellRoute(
        builder: (_, __, child) => NgoShell(child: child),
        routes: [
          GoRoute(path: '/ngo', builder: (_, __) => const NgoOffersScreen()),
          GoRoute(path: '/ngo/history', builder: (_, __) => const NgoHistoryScreen()),
          GoRoute(path: '/ngo/scan', builder: (_, __) => const NgoScanScreen()),
        ],
      ),

      // Logistics Role Routes
      ShellRoute(
        builder: (_, __, child) => LogisticsShell(child: child),
        routes: [
          GoRoute(path: '/logistics', builder: (_, __) => const LogisticsRoutesScreen()),
          GoRoute(path: '/logistics/scan', builder: (_, __) => const LogisticsScanScreen()),
        ],
      ),

      // Admin Role Routes
      ShellRoute(
        builder: (_, __, child) => AdminShell(child: child),
        routes: [
          GoRoute(path: '/admin', builder: (_, __) => const AdminDashboardScreen()),
          GoRoute(path: '/admin/models', builder: (_, __) => const AdminModelsScreen()),
          GoRoute(path: '/admin/users', builder: (_, __) => const AdminUsersScreen()),
          GoRoute(path: '/admin/audit', builder: (_, __) => const AdminAuditScreen()),
        ],
      ),
    ],
  );
});
