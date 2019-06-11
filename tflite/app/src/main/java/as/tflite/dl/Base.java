package as.tflite.dl;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import org.tensorflow.lite.Interpreter;

public class Base {
    protected Interpreter tfLite;

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    /**
     * Initializes a native TensorFlow session.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     */
    public Base ( final AssetManager assetManager,
                  final String modelFilename)
            throws IOException {
        try {
            tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
