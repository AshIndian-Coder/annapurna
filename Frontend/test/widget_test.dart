// Widget tests for the authentication entry points.
//
// The previous counter smoke test referenced a `MyApp` widget that no longer
// exists, so it could not compile and provided no coverage.

import 'package:annapurna/core/enums.dart';
import 'package:annapurna/data/dtos/models.dart';
import 'package:annapurna/features/auth/login_screen.dart';
import 'package:annapurna/features/auth/signup_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

Widget wrap(Widget child) => ProviderScope(
      child: MaterialApp(home: child),
    );

/// Scrolls the primary action into view before tapping it.
///
/// The auth screens are taller than the 800x600 test viewport, so a bare
/// `tap` silently misses the button (it stays at an off-screen offset) and the
/// form is never validated.
Future<void> submitForm(WidgetTester tester) async {
  await tester.ensureVisible(find.text('Create Account'));
  await tester.pumpAndSettle();
  await tester.tap(find.text('Create Account'));
  await tester.pumpAndSettle();
}

void main() {
  group('LoginScreen', () {
    testWidgets('offers a route to signup', (tester) async {
      await tester.pumpWidget(wrap(const LoginScreen()));

      expect(find.text('Annapurna'), findsOneWidget);
      expect(find.text('Sign In'), findsOneWidget);
      expect(find.widgetWithText(TextButton, 'Create Account'), findsOneWidget);
    });

    testWidgets('rejects an empty form without calling the API', (tester) async {
      await tester.pumpWidget(wrap(const LoginScreen()));

      await tester.tap(find.text('Sign In'));
      await tester.pump();

      expect(find.text('Please enter your email'), findsOneWidget);
      expect(find.text('Please enter your password'), findsOneWidget);
    });

    testWidgets('rejects a malformed email', (tester) async {
      await tester.pumpWidget(wrap(const LoginScreen()));

      await tester.enterText(find.byType(TextFormField).first, 'not-an-email');
      await tester.enterText(find.byType(TextFormField).last, 'sup3rSecret');
      await tester.tap(find.text('Sign In'));
      await tester.pump();

      expect(find.text('Enter a valid email address'), findsOneWidget);
    });
  });

  group('SignupScreen', () {
    testWidgets('asks for a name, email and password', (tester) async {
      await tester.pumpWidget(wrap(const SignupScreen()));

      expect(find.text('Create account'), findsOneWidget);
      expect(find.widgetWithText(TextFormField, 'Full name'), findsOneWidget);
      expect(find.widgetWithText(TextFormField, 'Email'), findsOneWidget);
      expect(find.text('Create Account'), findsOneWidget);
    });

    testWidgets('offers all four sign-up roles', (tester) async {
      await tester.pumpWidget(wrap(const SignupScreen()));

      expect(find.text('Kitchen'), findsOneWidget);
      expect(find.text('NGO'), findsOneWidget);
      expect(find.text('Driver'), findsOneWidget);
      expect(find.text('Admin'), findsOneWidget);
    });

    testWidgets('requires the passwords to match', (tester) async {
      await tester.pumpWidget(wrap(const SignupScreen()));

      final fields = find.byType(TextFormField);
      await tester.enterText(fields.at(0), 'Meera Kitchen');
      await tester.enterText(fields.at(1), 'meera@example.com');
      await tester.enterText(fields.at(2), 'sup3rSecret');
      await tester.enterText(fields.at(3), 'differentSecret');
      await submitForm(tester);

      expect(find.text('Passwords do not match'), findsOneWidget);
    });

    testWidgets('rejects a weak password', (tester) async {
      await tester.pumpWidget(wrap(const SignupScreen()));

      final fields = find.byType(TextFormField);
      await tester.enterText(fields.at(0), 'Meera Kitchen');
      await tester.enterText(fields.at(1), 'meera@example.com');
      await tester.enterText(fields.at(2), 'short');
      await tester.enterText(fields.at(3), 'short');
      await submitForm(tester);

      expect(find.text('Use at least 8 characters'), findsOneWidget);
    });
  });

  group('User.displayName', () {
    test('falls back to the email local part when the name is blank', () {
      final user = User(id: 'u1', name: '', email: 'meera@example.com', role: UserRole.kitchen);
      expect(user.displayName, 'meera');
    });

    test('falls back to the role when there is no name or email', () {
      final user = User(id: 'u1', name: '', email: '', role: UserRole.ngo);
      expect(user.displayName, 'NGO');
    });

    test('prefers a real name', () {
      final user = User(id: 'u1', name: 'Meera Kitchen', email: 'meera@example.com', role: UserRole.kitchen);
      expect(user.displayName, 'Meera Kitchen');
    });
  });

  group('User.fromJson', () {
    test('tolerates a payload with no name or email', () {
      final user = User.fromJson({'id': 'u1', 'role': 'NGO'});
      expect(user.id, 'u1');
      expect(user.name, '');
      expect(user.email, '');
      expect(user.role, UserRole.ngo);
    });
  });
}