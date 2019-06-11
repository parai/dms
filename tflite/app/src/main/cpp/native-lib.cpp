
#include <jni.h>
#include <string>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <android/log.h>
#include <android/bitmap.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "error", __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "debug", __VA_ARGS__))


using namespace cv;
using namespace std;

CascadeClassifier haar_face_cascade;

// context
Mat c_frame;
Mat c_grey;
vector<Rect> c_face_rect;

// from https://www.jianshu.com/p/08dcc910b088
void BitmapToMat2(JNIEnv *env, jobject& bitmap, Mat& mat, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &dst = mat;

    try {
        LOGD("nBitmapToMat");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

void BitmapToMat(JNIEnv *env, jobject& bitmap, Mat& mat) {
    BitmapToMat2(env, bitmap, mat, false);
}

void MatToBitmap2
        (JNIEnv *env, Mat& mat, jobject& bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &src = mat;

    // Creaing Bitmap Config Class
    jclass bmpCfgCls = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID bmpClsValueOfMid = env->GetStaticMethodID(bmpCfgCls, "valueOf",
            "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;");
    jobject jBmpCfg = env->CallStaticObjectMethod(bmpCfgCls, bmpClsValueOfMid,
            env->NewStringUTF("ARGB_8888"));

    // Creating a Bitmap Class
    jclass bmpCls = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMid = env->GetStaticMethodID(bmpCls, "createBitmap",
            "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    bitmap = env->CallStaticObjectMethod(bmpCls, createBitmapMid, src.cols, src.rows, jBmpCfg);

    try {
        LOGD("nMatToBitmap");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
                  info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                if (needPremultiplyAlpha)
                    cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else
                    src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}

void MatToBitmap(JNIEnv *env, Mat& mat, jobject& bitmap) {
    MatToBitmap2(env, mat, bitmap, false);
}

extern "C"
JNIEXPORT void
JNICALL
Java_as_tflite_MainActivity_asInit(
        JNIEnv *env,
        jobject /* this */) {
    haar_face_cascade.load("/sdcard/DMS/haarcascade_frontalface_alt.xml");
}


extern "C"
JNIEXPORT void
JNICALL
Java_as_tflite_MainActivity_asPreProcess(
        JNIEnv *env,
        jobject /* this */,
        jobject bitmap) {
    BitmapToMat(env, bitmap, c_frame);
    cvtColor(c_frame, c_grey, COLOR_BGR2GRAY);
    haar_face_cascade.detectMultiScale(c_grey, c_face_rect, 1.1, 5, 0);
}

extern "C"
JNIEXPORT int
JNICALL
Java_as_tflite_MainActivity_asGetFaceNumber(
        JNIEnv *env,
        jobject /* this */) {
     return c_face_rect.size();
}

extern "C"
JNIEXPORT jintArray
JNICALL
Java_as_tflite_MainActivity_asGetFaceRect(
        JNIEnv *env,
        jobject /* this */,
        int faceid) {
    jintArray result = env->NewIntArray(4);

    int ibuf[4] = {
        c_face_rect[faceid].x, c_face_rect[faceid].y,
        c_face_rect[faceid].width, c_face_rect[faceid].height
    };

     env->SetIntArrayRegion(result, 0, 4, ibuf);

    return result;
}

extern "C"
JNIEXPORT jobject
JNICALL
Java_as_tflite_MainActivity_asGetDebugBitmap(
        JNIEnv *env,
        jobject /* this */) {
    jobject bitmap;

    for(int i=0; i<c_face_rect.size(); i++)
    {
        rectangle(c_frame, c_face_rect[i], Scalar(0, 0, 255), 2);
    }

    MatToBitmap(env, c_frame, bitmap);

    return bitmap;
}


extern "C"
JNIEXPORT jobject
JNICALL
Java_as_tflite_MainActivity_asGetFaceBitmap(
        JNIEnv *env,
        jobject /* this */,
        int faceid) {
    jobject bitmap;

    Mat faceo(c_frame, c_face_rect[faceid]);
    Mat face;

    resize(faceo, face, Size(160,160), 0, 0, INTER_LINEAR);
    MatToBitmap(env, face, bitmap);

    return bitmap;
}



