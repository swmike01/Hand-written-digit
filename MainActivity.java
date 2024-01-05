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

import android.annotation.SuppressLint;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.divyanshu.draw.widget.DrawView;

public class MainActivity extends AppCompatActivity {

    private DrawView drawView;
    private Button clearButton;
    private TextView predictedTextView;
    private org.tensorflow.lite.codelabs.digitclassifier.DigitClassifier digitClassifier;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Setup view instances.
        drawView = findViewById(R.id.draw_view);
        drawView.setStrokeWidth(70.0f);
        drawView.setColor(Color.WHITE);
        drawView.setBackgroundColor(Color.BLACK);
        clearButton = findViewById(R.id.clear_button);
        predictedTextView = findViewById(R.id.predicted_text);

        // Setup clear drawing button.
        clearButton.setOnClickListener(view -> {
            drawView.clearCanvas();
            predictedTextView.setText(getString(R.string.prediction_text_placeholder));
        });

        // Setup classification trigger so that it classifies after every stroke drawn.
        drawView.setOnTouchListener((view, event) -> {
            // As we have interrupted DrawView's touch event,
            // we first need to pass touch events through to the instance for the drawing to show up.
            drawView.onTouchEvent(event);

            // Then if the user finished a touch event, run classification
            if (event.getAction() == MotionEvent.ACTION_UP) {
                classifyDrawing();
            }

            return true;
        });

        // Setup digit classifier.
        digitClassifier = new org.tensorflow.lite.codelabs.digitclassifier.DigitClassifier(this);
        digitClassifier.initialize()
                .addOnFailureListener(e -> Log.e(TAG, "Error setting up digit classifier.", e));
    }

    @Override
    protected void onDestroy() {
        // Sync DigitClassifier instance lifecycle with MainActivity lifecycle,
        // and free up resources (e.g., TF Lite instance) once the activity is destroyed.
        digitClassifier.close();
        super.onDestroy();
    }

    private void classifyDrawing() {
        if (drawView != null && digitClassifier.isInitialized()) {
            digitClassifier.classifyAsync(drawView.getBitmap())
                    .addOnSuccessListener(resultText -> predictedTextView.setText(resultText))
                    .addOnFailureListener(e -> {
                        predictedTextView.setText(getString(R.string.classification_error_message, e.getLocalizedMessage()));
                        Log.e(TAG, "Error classifying drawing.", e);
                    });
        }
    }

    private static final String TAG = "MainActivity";
}
