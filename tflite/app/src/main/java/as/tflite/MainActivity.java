/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package as.tflite;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.RectF;

import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Size;
import as.tflite.customview.OverlayView;
import as.tflite.env.Logger;
import as.tflite.env.ImageUtils;

import as.tflite.customview.OverlayView.DrawCallback;
import as.tflite.tracking.MultiBoxTracker;
import as.tflite.tracking.Recognition;

import as.tflite.dl.FaceNet;


import android.content.res.AssetManager;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap rotatedBitmap = null;
    private Bitmap debugBitmap = null;

    private Matrix frameToRotateTransform;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private MultiBoxTracker tracker;

    private Lock asMutex;

    private FaceNet facenet;

    public native void asInit();
    public native void asPreProcess(Bitmap img);
    public native int  asGetFaceNumber();
    public native int[]  asGetFaceRect(int faceid);
    public native Bitmap asGetFaceBitmap(int faceid);

    public native Bitmap asGetDebugBitmap();

    static {
        System.loadLibrary("native-lib");
    }

    private void copyAssetFileIfNotExist(String filename) {
        AssetManager assetManager = getAssets();
        String newFileName = "/sdcard/DMS/" + filename;
        File file = new File(newFileName);
        if (file.exists()) {
            LOGGER.i("File "+newFileName+" already exist.");
            return;
        }
        file.mkdirs();
        try {
            InputStream in = assetManager.open(filename);
            OutputStream out = new FileOutputStream(newFileName);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            in.close();
            out.flush();
            out.close();
        } catch (Exception e) {
            LOGGER.e(e.getMessage());
        }
    }

    private boolean hasPermission(final String permission) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission(final String permission) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(permission)) {
                Toast.makeText(
                        MainActivity.this,
                        permission+" permission is required for this demo",
                        Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[] {permission}, 1);
        }
    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!hasPermission(Manifest.permission.READ_EXTERNAL_STORAGE)) {
            requestPermission(Manifest.permission.READ_EXTERNAL_STORAGE);
        }

        if (!hasPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            requestPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        copyAssetFileIfNotExist("haarcascade_frontalface_alt.xml");
        asMutex = new ReentrantLock(false);;
        asInit();
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        // init DL models
        AssetManager assetManager = getAssets();
        try {
            facenet = new FaceNet(assetManager, "facenet.tflite");
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing FaceNet!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "FaceNet could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        tracker = new MultiBoxTracker(this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        rotatedBitmap = Bitmap.createBitmap(previewHeight, previewWidth, Config.ARGB_8888);

        frameToRotateTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        previewHeight, previewWidth,
                        -sensorOrientation, MAINTAIN_ASPECT);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
            new DrawCallback() {
                @Override
                public void drawCallback(final Canvas canvas) {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                }
            });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        final Canvas canvas = new Canvas(rotatedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToRotateTransform, null);

        readyForNextImage();

        runInBackground(
            new Runnable() {
                @Override
                public void run() {
                    LOGGER.i("Running detection on image " + currTimestamp);
                    final long startTime = SystemClock.uptimeMillis();

                    final List<Recognition> mappedRecognitions =
                            new LinkedList<Recognition>();

                    asMutex.lock();
                    asPreProcess(rotatedBitmap);

                    int faceNumber = asGetFaceNumber();
                    for(int i=0; i<faceNumber; i++) {
                        Bitmap face = asGetFaceBitmap(i);
                        float dis = facenet.predict(face);
                        if(dis == 0.0) {
                            debugBitmap = face;
                        }

                        int[] rect = asGetFaceRect(i);
                        int x=rect[0], y=rect[1], w=rect[2], h=rect[3];
                        final RectF rectangle = new RectF(y, x, y+h, x+w);
                        Recognition rec = new Recognition(""+i,"face"+i, dis, rectangle);
                        mappedRecognitions.add(rec);
                    }

                    //debugBitmap = asGetDebugBitmap();
                    asMutex.unlock();

                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                    tracker.trackResults(mappedRecognitions, currTimestamp);
                    trackingOverlay.postInvalidate();

                    computingDetection = false;

                    runOnUiThread(
                        new Runnable() {
                            @Override
                            public void run() {
                                showFrameInfo(previewWidth + "x" + previewHeight);
                                showInference(lastProcessingTimeMs + "ms");
                                if(debugBitmap != null) {
                                    debugImageView.setImageBitmap(debugBitmap);
                                    debugBitmap = null;
                                }
                            }
                        });
                }
            });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {

    }

    @Override
    protected void setNumThreads(final int numThreads) {

    }
}
