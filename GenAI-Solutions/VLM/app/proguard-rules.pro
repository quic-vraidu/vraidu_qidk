# Add project specific ProGuard rules here.
# Keep Genie JNI classes
-keep class com.qualcomm.qidk.vlm.** { *; }
-keepclassmembers class * {
    native <methods>;
}
