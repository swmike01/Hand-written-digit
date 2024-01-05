/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.codelabs.digitclassifier;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.TaskCompletionSource;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DigitClassifier {

    private final Context context;
    private Interpreter interpreter;
    private boolean isInitialized = false;

    private final ExecutorService executorService = Executors.newCachedThreadPool();

    private int inputImageWidth = 0; // will be inferred from TF Lite model.
    private int inputImageHeight = 0; // will be inferred from TF Lite model.
    private int modelInputSize = 0; // will be inferred from TF Lite model.

    public DigitClassifier(Context context) {
        this.context = context;
    }

    public Task<Void> initialize() {
        TaskCompletionSource<Void> task = new TaskCompletionSource<>();
        executorService.execute(() -> {
            try {
                initializeInterpreter();
                task.setResult(null);
            } catch (IOException e) {
                task.setException(e);
            }
        });
        return task.getTask();
    }

    private void initializeInterpreter() throws IOException {
        AssetManager assetManager = context.getAssets();
        ByteBuffer model = loadModelFile(assetManager, "mnist.tflite");
        interpreter = new Interpreter(model);
        int[] inputShape = interpreter.getInputTensor(0).shape();
        inputImageWidth = inputShape[1];
        inputImageHeight = inputShape[2];
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * PIXEL_SIZE;

        isInitialized = true;
        Log.d(TAG, "Initialized TFLite interpreter.");
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String filename) throws IOException {
        FileInputStream inputStream = new FileInputStream(assetManager.openFd(filename).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = assetManager.openFd(filename).getStartOffset();
        long declaredLength = assetManager.openFd(filename).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Task<String> classifyAsync(Bitmap bitmap) {
        TaskCompletionSource<String> task = new TaskCompletionSource<>();
        executorService.execute(() -> {
            String result = classify(bitmap);
            task.setResult(result);
        });
        return task.getTask();
    }

    private String classify(Bitmap bitmap) {
        checkIsInitialized();

        Bitmap resizedImage = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedImage);

        float[][] output = new float[1][OUTPUT_CLASSES_COUNT];
        interpreter.run(byteBuffer, output);

        int maxIndex = getMaxIndex(output[0]);
        float confidence = output[0][maxIndex];
        return String.format("Prediction Result: %d\nConfidence: %2f", maxIndex, confidence);
    }

    private int getMaxIndex(float[] array) {
        int maxIndex = -1;
        float maxVal = Float.MIN_VALUE;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void close() {
        executorService.execute(() -> {
            if (interpreter != null) {
                interpreter.close();
                Log.d(TAG, "Closed TFLite interpreter.");
            }
        });
    }

    private void checkIsInitialized() {
        if (!isInitialized) {
            throw new IllegalStateException("TF Lite Interpreter is not initialized yet.");
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelInputSize);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int pixelValue : pixels) {
            int r = (pixelValue >> 16) & 0xFF;
            int g = (pixelValue >> 8) & 0xFF;
            int b = pixelValue & 0xFF;

            float normalizedPixelValue = (r + g + b) / 3.0f / 255.0f;
            byteBuffer.putFloat(normalizedPixelValue);
        }

        return byteBuffer;
    }

    private static final String TAG = "DigitClassifier";
    private static final int FLOAT_TYPE_SIZE = 4;
    private static final int PIXEL_SIZE = 1;
    private static final int OUTPUT_CLASSES_COUNT = 10;

    public boolean isInitialized() {
        return isInitialized;
    }
}
