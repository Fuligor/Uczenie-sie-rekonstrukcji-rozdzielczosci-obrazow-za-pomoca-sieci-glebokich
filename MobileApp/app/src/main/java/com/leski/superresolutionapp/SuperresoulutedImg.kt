package com.leski.superresolutionapp


import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.AsyncTask
import android.os.Bundle
import android.os.StrictMode
import android.provider.MediaStore
import android.util.Base64
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.leski.superresolutionapp.databinding.ActivitySuperresoulutedImgBinding
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.*
import java.net.NoRouteToHostException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

class SuperresoulutedImg : AppCompatActivity() {
    private lateinit var binding: ActivitySuperresoulutedImgBinding
    private var itemShouldBeEnabled: Boolean = false

    fun interface Listener {
        fun onImageUpscaled(bitmap: Bitmap?)
    }

    private class UpscaleImageTask(val resolver: ContentResolver, var listener: Listener): AsyncTask<Bundle, Void, Bitmap>() {
        lateinit var imgUri: Uri
        lateinit var serverIp: String
        lateinit var model: String

        override fun doInBackground(vararg params: Bundle?): Bitmap {
            val extras = params[0]!!

            imgUri = extras.get("imgURI") as Uri
            serverIp = extras.get("IP") as String
            model = extras.get("Model") as String

            val bitmap = MediaStore.Images.Media.getBitmap(resolver, imgUri)

            val postUrl = "http://$serverIp:8080/"
            val postBodyText = getStringImage(bitmap)
            val mediaType = "text/plain; charset=utf-8".toMediaType()
            val postBody = postBodyText.toRequestBody(mediaType)

            return postRequest(postUrl, postBody)
        }

        override fun onPostExecute(result: Bitmap?) {
            if (result != null) {
                listener.onImageUpscaled(result);
            }
        }

        private fun getStringImage(bitmap: Bitmap):String {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos)

            val imageBytes = baos.toByteArray()
            return Base64.encodeToString(imageBytes, Base64.DEFAULT)
        }

        private fun postRequest(postUrl: String, postBody: RequestBody): Bitmap {
            val client = OkHttpClient.Builder()
                .readTimeout(7, TimeUnit.MINUTES)
                .build()

            val request = Request.Builder()
                .url(postUrl)
                .header("Model", model)
                .post(postBody)
                .build()

            val response =  client.newCall(request).execute()

            if(response.isSuccessful) {
                val imageResult = response.body?.string()
                val imageBytes = Base64.decode(imageResult, 0)

                return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            }
            else {
                throw NoRouteToHostException()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySuperresoulutedImgBinding.inflate(layoutInflater)
        val view = binding.root
        val extras = intent.extras ?: return

        setContentView(view)
        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()

        StrictMode.setThreadPolicy(policy)

        binding.imageView.setImageURI(null)
        binding.progressBar.visibility = View.VISIBLE

        val listener = Listener {


            runOnUiThread {
                try {
                    binding.progressBar.visibility = View.GONE
                    binding.imageView.setImageBitmap(it)
                    itemShouldBeEnabled = true
                    invalidateOptionsMenu()

                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }

        UpscaleImageTask(this.contentResolver, listener).execute(extras)
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.bar, menu)

        return true
    }

    override fun onPrepareOptionsMenu(menu: Menu): Boolean {
        val saveButt = menu.findItem(R.id.miSave)
        val rejectButt = menu.findItem(R.id.miReject)
        if (itemShouldBeEnabled) {
            saveButt.isEnabled = true
            rejectButt.isEnabled = true
            saveButt.icon.alpha = 255
            rejectButt.icon.alpha = 255
        } else {
            // disabled
            saveButt.isEnabled = false
            rejectButt.isEnabled = false
            saveButt.icon.alpha = 130
            rejectButt.icon.alpha = 130
        }
        return super.onPrepareOptionsMenu(menu);
    }
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {

            R.id.miReject -> {
                super.onBackPressed()
                true
            }
            R.id.miSave -> {
                val bitmap = (binding.imageView.getDrawable() as BitmapDrawable).bitmap
                val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
                val savedImageURL = MediaStore.Images.Media.insertImage(
                    this.contentResolver,
                    bitmap,
                    "JPEG_${timeStamp}_",
                    "JPEG_${timeStamp}_"
                )
                Toast.makeText(this, "Zdjęcie zostało zapisane", Toast.LENGTH_SHORT).show()
                super.onBackPressed()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}