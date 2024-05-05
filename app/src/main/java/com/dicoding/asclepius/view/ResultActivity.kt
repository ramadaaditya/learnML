package com.dicoding.asclepius.view

import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.dicoding.asclepius.R
import com.dicoding.asclepius.databinding.ActivityResultBinding
import com.dicoding.asclepius.helper.ImageClassifierHelper
import org.tensorflow.lite.task.vision.classifier.Classifications

class ResultActivity : AppCompatActivity() {
    private lateinit var binding: ActivityResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val imageUriString = intent.getStringExtra(IMAGE_URI)
        if (imageUriString != null) {
            val imageUri = Uri.parse(imageUriString)
            displayImage(imageUri)

            val imageClassifierHelper = ImageClassifierHelper(
                context = this,
                classifierListener = object : ImageClassifierHelper.ClassifierListener {
                    override fun onError(errorMessage: String) {
                        Log.d(TAG, "Error: $errorMessage")
                    }
                    override fun onResult(result: List<Classifications>?, interfaceTime: Long) {
                        result?.let { showResult(it) }
                    }
                }
            )
            imageClassifierHelper.classifyStaticImage(imageUri)
        } else {
            Log.e(TAG, "No image URI provided")
            finish()
        }
    }

    private fun showResult(result: List<Classifications>?) {
        result?.takeIf { it.isNotEmpty() }?.get(0)?.let { topResult ->
            val label = topResult.categories[0].label
            val score = topResult.categories[0].score

            fun Float.formatToString(): String {
                return String.format("%.2f%%", this * 100)
            }
            "$label ${score.formatToString()}".also { binding.resultText.text = it }
        } ?: run {
            binding.resultText.text = getString(R.string.no_result_available)
        }
    }


    private fun displayImage(uri: Uri) {
        Log.d(TAG, "displayImage: $uri")
        binding.resultImage.setImageURI(uri)
    }

    companion object {
        const val IMAGE_URI = "img_uri"
        const val TAG = "imagePicker"
    }
}