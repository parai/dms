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

package as.tflite.customview;

import android.content.Context;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.view.View;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;

/** A simple View providing a render callback to other classes. */
public class OverlayView extends View {

  private Paint paint;
  private Rect rectangle;

  public OverlayView(final Context context, final AttributeSet attrs) {
    super(context, attrs);
    paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(2.0f);

    int x = 100;
    int y = 100;
    int sideLength = 200;

    // create a rectangle that we'll draw later
    rectangle = new Rect(x, y, sideLength, sideLength);
  }

  @Override
  protected void onDraw(Canvas canvas) {
    canvas.drawRect(rectangle, paint);
  }
}
