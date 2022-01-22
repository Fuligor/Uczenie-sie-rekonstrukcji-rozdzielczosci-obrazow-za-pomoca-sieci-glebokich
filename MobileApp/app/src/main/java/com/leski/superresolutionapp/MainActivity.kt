package com.leski.superresolutionapp


import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.drawable.BitmapDrawable

import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Patterns
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*



class MainActivity : AppCompatActivity() {
    private lateinit var dropdown: Spinner
    private lateinit var ipnumber: EditText
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        dropdown = findViewById(R.id.spinner1)
        ipnumber = findViewById(R.id.ipnumber)

        val items = arrayOf("UNet", "KPNLP", "MZSR_bicubic", "MZSR_kernelGAN")

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, items)
        dropdown.adapter = adapter

    }

    lateinit var currentPhotoPath: String

    @Throws(IOException::class)
    private fun createImageFile(): File {

        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_",
            ".jpg",
            storageDir
        ).apply {

            currentPhotoPath = absolutePath
        }
    }


    val REQUEST_IMAGE_CAPTURE = 1
    fun dispatchTakePictureIntent(v: View) {
        if(Patterns.IP_ADDRESS.matcher(ipnumber.text.toString()).matches())
        {
            Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
                takePictureIntent.resolveActivity(packageManager)?.also {
                    val photoFile: File? = try {
                        createImageFile()
                    } catch (ex: IOException) {
                        Toast.makeText(this, R.string.cameraOpenError, Toast.LENGTH_SHORT).show()
                        null
                    }
                    photoFile?.also {
                        val photoURI: Uri = FileProvider.getUriForFile(
                            this,
                            "com.leski.superresolutionapp.fileprovider",
                            it
                        )
                        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                    }
                }
            }
        }
        else{
            Toast.makeText(this, "Niepoprawny adres IP", Toast.LENGTH_SHORT).show()
        }

    }

    val REQUEST_IMAGE_FROM_GALLERY = 2
    fun openGalleryForImage(v: View) {
        if(Patterns.IP_ADDRESS.matcher(ipnumber.text.toString()).matches())
        {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            try {
                startActivityForResult(intent, REQUEST_IMAGE_FROM_GALLERY)
            } catch (e: ActivityNotFoundException) {
                Toast.makeText(this, R.string.galleryOpenError, Toast.LENGTH_SHORT).show()
            }
        }
        else {
            Toast.makeText(this, "Niepoprawny adres IP", Toast.LENGTH_SHORT).show()
        }

    }
    private fun galleryAddPic() {
        Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE).also { mediaScanIntent ->
            val f = File(currentPhotoPath)
            mediaScanIntent.data = Uri.fromFile(f)
            sendBroadcast(mediaScanIntent)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            val intent = Intent(this, SuperresoulutedImg::class.java)
            intent.putExtra("Model", dropdown.selectedItem.toString())
            intent.putExtra("IP", ipnumber.text.toString())

            if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
                galleryAddPic()

                val f = File(currentPhotoPath)
                intent.putExtra("imgURI", Uri.fromFile(f))
                startActivity(intent)
            }
            if (resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_FROM_GALLERY) {
                val pom = ImageView(this.applicationContext)
                pom.setImageURI(data?.data)

                val imageBitmap =
                    (pom.drawable as BitmapDrawable).bitmap

                intent.putExtra("imgURI", data?.data)
                startActivity(intent)
            }
        }
    }
}