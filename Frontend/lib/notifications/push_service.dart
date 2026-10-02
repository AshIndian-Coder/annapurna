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
  final FirebaseMessaging _messaging = FirebaseMessaging.instance;
  final FlutterLocalNotificationsPlugin _localNotifications = FlutterLocalNotificationsPlugin();

  PushService(this._api);

  Future<void> initialize() async {
    try {
      await Firebase.initializeApp();
    } catch (_) {}

    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

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

    final settings = await _messaging.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );

    if (settings.authorizationStatus == AuthorizationStatus.authorized) {
      final token = await _messaging.getToken();
      if (token != null) {
        await _registerToken(token);
      }

      _messaging.onTokenRefresh.listen(_registerToken);

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
  }

  Future<void> _registerToken(String token) async {
    await _api.post('/users/fcm-token', data: {'token': token});
  }
}
