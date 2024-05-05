package com.dicoding.asclepius.helper

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import com.dicoding.asclepius.R
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.lang.IllegalStateException

@Suppress("DEPRECATION")
class ImageClassifierHelper(
    private var threshold: Float = 0.1f,
    private var maxResult: Int = 3,
    private val modelName: String = "cancer_classification.tflite",
    private val context: Context,
    private val classifierListener: ClassifierListener?

) {
    interface ClassifierListener {
        fun onError(errorMessage: String)
        fun onResult(
            result: List<Classifications>?,
            interfaceTime: Long
        )
    }

    private var imageClassifier: ImageClassifier? = null

    init {
        setupImageClassifier()
    }

    private fun setupImageClassifier() {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResult)
        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(4)
        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        try {
            imageClassifier = ImageClassifier.createFromFileAndOptions(
                context,
                modelName,
                optionsBuilder.build()
            )
        } catch (e: IllegalStateException) {
            classifierListener?.onError(context.getString(R.string.image_classifier_failed))
            Log.e(TAG, e.message.toString())
        }
    }

    fun classifyStaticImage(imageUri: Uri) {
        if (imageClassifier == null) {
            setupImageClassifier()
        }
        val bitmap = getImageBitmap(imageUri)
        val tensorImage = preprocessImage(bitmap)
        val result = performInference(tensorImage)
        notifyResults(result)
    }

    private fun getImageBitmap(imageUri: Uri): Bitmap {
        return if (checkVersion()) {
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            ImageDecoder.decodeBitmap(source)
        } else {
            MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun preprocessImage(bitmap: Bitmap): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeMethod.NEAREST_NEIGHBOR))
            .add(CastOp(DataType.FLOAT32))
            .build()
        return imageProcessor.process(TensorImage.fromBitmap(bitmap))
    }

    private fun checkVersion(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
    }

    private fun performInference(tensorImage: TensorImage): List<Classifications>? {
        var inferenceTime = SystemClock.uptimeMillis()
        val results = imageClassifier?.classify(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        classifierListener?.onResult(results, inferenceTime)
        return results
    }

    private fun notifyResults(result: List<Classifications>?) {
        classifierListener?.onResult(result, 0)
    }

    companion object {
        private const val TAG = "ImageClassifierHelper"
    }
}