import 'dart:convert';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/api/api_client.dart';

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  try {
    await Firebase.initializeApp();
  } catch (_) {}
}

final pushServiceProvider = Provider<PushService>((ref) {
  return PushService(ref.read(apiClientProvider));
});

class PushService {
  final ApiClient _api;
  FirebaseMessaging? _messaging;
  final FlutterLocalNotificationsPlugin _localNotifications = FlutterLocalNotificationsPlugin();

  PushService(this._api) {
    try {
      // Only instantiate FirebaseMessaging if Firebase is initialized
      if (Firebase.apps.isNotEmpty) {
        _messaging = FirebaseMessaging.instance;
      }
    } catch (_) {}
  }

  Future<void> initialize() async {
    try {
      if (Firebase.apps.isEmpty) {
        await Firebase.initializeApp();
      }

      _messaging ??= FirebaseMessaging.instance;
      FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
    } catch (_) {
      print("Skipping Firebase initialization in push service.");
    }

    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings();
    const initSettings = InitializationSettings(android: androidSettings, iOS: iosSettings);

    await _localNotifications.initialize(
      initSettings,
      onDidReceiveNotificationResponse: (details) {
        if (details.payload != null) {
          final data = jsonDecode(details.payload!);
        }
      },
    );

    const channel = AndroidNotificationChannel(
      'high_importance_channel',
      'High Importance Notifications',
      importance: Importance.max,
    );

    await _localNotifications
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(channel);

    try {
      if (_messaging == null) return;

      final settings = await _messaging!.requestPermission(
        alert: true,
        badge: true,
        sound: true,
      );

      if (settings.authorizationStatus == AuthorizationStatus.authorized) {
        final token = await _messaging!.getToken();
        if (token != null) {
          await _registerToken(token);
        }

        _messaging!.onTokenRefresh.listen(_registerToken);

        FirebaseMessaging.onMessage.listen((RemoteMessage message) {
          final notification = message.notification;
          final android = message.notification?.android;

          if (notification != null && android != null) {
            _localNotifications.show(
              notification.hashCode,
              notification.title,
              notification.body,
              NotificationDetails(
                android: AndroidNotificationDetails(
                  channel.id,
                  channel.name,
                  icon: android.smallIcon,
                  importance: Importance.max,
                  priority: Priority.max,
                ),
              ),
              payload: jsonEncode(message.data),
            );
          }
        });
      }
    } catch (_) {
      print('Firebase push notifications skipped.');
    }
  }

  Future<void> _registerToken(String token) async {
    // Contract #37: POST /devices — register FCM push token.
    await _api.post('/devices', data: {
      'platform': 'ANDROID',
      'token': token,
      'app_version': appVersion,
    }, parser: (data) => data);
  }
}
