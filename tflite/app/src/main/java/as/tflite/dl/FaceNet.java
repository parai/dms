package as.tflite.dl;

import android.content.res.AssetManager;

import android.graphics.Bitmap;

import java.io.IOException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import java.lang.Math;

import as.tflite.dl.Base;
import as.tflite.env.Logger;

public class FaceNet extends Base {

    private static final Logger LOGGER = new Logger();

    private ByteBuffer input;
    private float[][] output;

    private float[][] last_output = null;

    private int[] intValues;

    public FaceNet(final AssetManager assetManager,
                   final String modelFilename)
            throws IOException {
        super(assetManager, modelFilename);

        intValues = new int[160 * 160];

        input = ByteBuffer.allocateDirect(1 * 160 * 160 * 3 * 4);
        input.order(ByteOrder.nativeOrder());
        output = new float[1][512];
    }

    public float predict(Bitmap face) {
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        face.getPixels(intValues, 0, face.getWidth(), 0, 0, face.getWidth(), face.getHeight());

        float mean = 0.0f;
        for (int i = 0; i < 160; ++i) {
            for (int j = 0; j < 160; ++j) {
                int pixelValue = intValues[i * 160 + j];
                int r = (pixelValue >> 16) & 0xFF;
                int g = (pixelValue >> 8) & 0xFF;
                int b = pixelValue & 0xFF;
                mean += r + g + b;
            }
        }
        mean = mean/(160*160*3);

        float std = 0.0f;
        for (int i = 0; i < 160; ++i) {
            for (int j = 0; j < 160; ++j) {
                int pixelValue = intValues[i * 160 + j];
                int r = (pixelValue >> 16) & 0xFF;
                int g = (pixelValue >> 8) & 0xFF;
                int b = pixelValue & 0xFF;
                std += (r-mean)*(r-mean) + (g-mean)*(g-mean) + (b-mean)*(b-mean);
            }
        }
        std = std/(160*160*3);
        std = (float)Math.max(Math.sqrt(std), 1.0f / Math.sqrt(160*160*3));
        LOGGER.i("std=%.2f, mean=%.2f", std, mean);

        input.rewind();
        for (int i = 0; i < 160; ++i) {
            for (int j = 0; j < 160; ++j) {
                int pixelValue = intValues[i * 160 + j];
                int r = (pixelValue >> 16) & 0xFF;
                int g = (pixelValue >> 8) & 0xFF;
                int b = pixelValue & 0xFF;
                input.putFloat((r - mean) / std);
                input.putFloat((g - mean) / std);
                input.putFloat((b - mean) / std);
            }
        }

        Object[] inputArray = {input};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, output);

        // Run the inference call.
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        float dis = 0.0f;
        if(last_output == null) {
            last_output = new float[1][512];

            for(int i=0; i<512; i++) {
                last_output[0][i] = output[0][i];
            }
        } else {
            dis = euclidean_distances(last_output, output);

            LOGGER.i("distance is %.2f", dis);
        }
        
        return dis;
    }


    private float euclidean_distances(float a[][], float b[][]) {
        float dis = 0.0f;

        for(int i=0; i<512; i++) {
            float delta = a[0][i]-b[0][i];
            dis += delta*delta;
        }

        return dis;
    }
}
