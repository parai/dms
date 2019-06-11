package as.tflite.dl;

import android.content.res.AssetManager;

import android.graphics.Bitmap;

import java.io.IOException;

import as.tflite.dl.Base;
import as.tflite.env.Logger;

public class FaceNet extends Base {

    private static final Logger LOGGER = new Logger();

    public FaceNet(final AssetManager assetManager,
                   final String modelFilename)
            throws IOException {
        super(assetManager, modelFilename);
    }

    public void predict(Bitmap face) {

    }
}
