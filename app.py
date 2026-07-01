import os
import json
import traceback
import h5py
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

MODEL_PATH = r'D:\vm\comvis\Uas_ArryComvis\garbage_classifier_mobilenetv2.h5'
model = None
input_shape = (224, 224)  # Default MobileNetV2 shape
classes = []

# Intelligent class mapping helper based on output size
# You can customize these lists to match your exact dataset labels.
CLASS_MAPPINGS = {
    2: [
        {'name': 'Organik', 'desc': 'Sampah organik seperti sisa makanan, daun, dan buah yang mudah terurai.', 'recycle_tip': 'Dapat dijadikan pupuk kompos atau pakan ternak.'},
        {'name': 'Anorganik', 'desc': 'Sampah anorganik seperti plastik, kaca, logam yang tidak mudah terurai.', 'recycle_tip': 'Pisahkan dan bersihkan sebelum disalurkan ke bank sampah atau tempat daur ulang.'}
    ],
    3: [
        {'name': 'Organik', 'desc': 'Sampah organik seperti daun, makanan, dan bahan alami lainnya.', 'recycle_tip': 'Komposkan untuk menyuburkan tanaman.'},
        {'name': 'Anorganik', 'desc': 'Sampah plastik, kaleng, dan kaca.', 'recycle_tip': 'Kumpulkan dan salurkan ke bank sampah.'},
        {'name': 'B3 (Bahan Berbahaya)', 'desc': 'Sampah beracun seperti baterai, lampu, obat kedaluwarsa, atau elektronik.', 'recycle_tip': 'Buang di tempat penampungan khusus sampah B3, jangan dicampur dengan sampah rumah tangga.'}
    ],
    4: [
        {'name': 'Organik', 'desc': 'Sampah dari sisa makanan dan bahan organik hayati.', 'recycle_tip': 'Komposkan secara mandiri di rumah.'},
        {'name': 'Anorganik', 'desc': 'Sampah kering non-organik seperti plastik, kertas bersih, kaca.', 'recycle_tip': 'Kumpulkan untuk didaur ulang.'},
        {'name': 'B3 (Bahan Berbahaya)', 'desc': 'Baterai, neon bekas, sisa pestisida, aerosol.', 'recycle_tip': 'Serahkan ke pos pengumpulan limbah B3 terdekat.'},
        {'name': 'Residu', 'desc': 'Sampah sisa yang sulit didaur ulang seperti popok sekali pakai, tisu basah, pembalut.', 'recycle_tip': 'Buang ke TPA dengan dibungkus rapi.'}
    ],
    10: [
        {
            'name': 'Baterai (Battery)',
            'desc': 'Baterai bekas mengandung bahan kimia beracun seperti timbal, merkuri, dan kadmium.',
            'recyclable': False,
            'recyclable_text': 'Tidak Layak Didaur Ulang (Limbah B3)',
            'recycle_tip': 'Jangan membuang baterai bekas ke tempat sampah biasa. Kumpulkan secara terpisah dan serahkan ke tempat pembuangan khusus limbah B3 (Bahan Berbahaya dan Beracun) terdekat.'
        },
        {
            'name': 'Sampah Organik (Biological)',
            'desc': 'Sisa makanan, daun gugur, sisa sayuran, buah-buahan, dan bahan organik alami lainnya.',
            'recyclable': False,
            'recyclable_text': 'Dapat Dikomposkan (Bukan Daur Ulang Material)',
            'recycle_tip': 'Kelola sampah organik menjadi pupuk kompos di rumah untuk menyuburkan tanah tanaman, atau salurkan ke pengolahan kompos lokal.'
        },
        {
            'name': 'Kardus (Cardboard)',
            'desc': 'Karton tebal, kotak kemasan makanan kering, dan kardus pembungkus paket.',
            'recyclable': True,
            'recyclable_text': 'Layak Didaur Ulang (Recyclable)',
            'recycle_tip': 'Lepaskan selotip/lakban yang menempel, ratakan atau lipat kardus hingga pipih agar hemat tempat, lalu simpan di area kering sebelum disalurkan ke bank sampah.'
        },
        {
            'name': 'Pakaian & Tekstil (Clothes)',
            'desc': 'Baju bekas, kaos kaki, kain celana, kain perca, atau bahan tekstil bekas pakai lainnya.',
            'recyclable': True,
            'recyclable_text': 'Layak Didaur Ulang / Disumbangkan',
            'recycle_tip': 'Pakaian layak pakai sebaiknya didonasikan. Pakaian rusak dapat disalurkan ke organisasi daur ulang serat tekstil untuk diolah kembali menjadi benang atau kain lap industri.'
        },
        {
            'name': 'Kaca (Glass)',
            'desc': 'Botol beling, toples kaca, cermin rusak, gelas kaca bekas wadah makanan/minuman.',
            'recyclable': True,
            'recyclable_text': 'Layak Didaur Ulang (Recyclable)',
            'recycle_tip': 'Bilas wadah kaca dari sisa makanan/minuman. Kumpulkan secara aman dan hati-hati agar tidak melukai petugas kebersihan saat diangkut.'
        },
        {
            'name': 'Logam & Kaleng (Metal)',
            'desc': 'Kaleng minuman alumunium, kaleng kaldu, paku, kawat besi, atau barang logam bekas lainnya.',
            'recyclable': True,
            'recyclable_text': 'Layak Didaur Ulang (Recyclable)',
            'recycle_tip': 'Bilas sisa minuman/makanan dari dalam kaleng, tekan kaleng hingga agak pipih agar hemat ruang penyimpanan sampah daur ulang Anda.'
        },
        {
            'name': 'Kertas (Paper)',
            'desc': 'Koran bekas, majalah, selebaran kertas, amplop surat, buku tulis tua, atau kertas kantor.',
            'recyclable': True,
            'recyclable_text': 'Layak Didaur Ulang (Recyclable)',
            'recycle_tip': 'Jaga kertas agar tetap kering dan bersih dari noda minyak atau sisa makanan yang dapat merusak kualitas serat daur ulang kertas.'
        },
        {
            'name': 'Plastik (Plastic)',
            'desc': 'Botol plastik, kemasan detergen, wadah makanan plastik, gelas plastik, atau jeriken.',
            'recyclable': True,
            'recyclable_text': 'Layak Didaur Ulang (Recyclable)',
            'recycle_tip': 'Bilas botol/wadah plastik dari sisa isinya, lepaskan tutup botol serta label kertas, lalu remas botol untuk meminimalkan volume penyimpanan.'
        },
        {
            'name': 'Sepatu (Shoes)',
            'desc': 'Sepatu bekas, sandal karet, atau alas kaki dari bahan kulit/sintetis lainnya.',
            'recyclable': False,
            'recyclable_text': 'Tidak Layak Didaur Ulang (Bahan Komposit)',
            'recycle_tip': 'Alas kaki bekas sangat sulit didaur ulang secara massal karena tersusun dari campuran lem, karet, busa, dan kain. Jika masih bagus, disarankan untuk disumbangkan.'
        },
        {
            'name': 'Sampah Residu (Trash)',
            'desc': 'Sampah rumah tangga harian yang kotor atau berlapis bahan campuran (sachet makanan berlapis alumunium, tisu basah kotor, popok sekali pakai).',
            'recyclable': False,
            'recyclable_text': 'Tidak Layak Didaur Ulang (Residu Umum)',
            'recycle_tip': 'Bungkus rapat sampah residu ini dan buang ke tempat sampah umum untuk dikirim secara langsung ke Tempat Pembuangan Akhir (TPA).'
        }
    ]
}

def load_keras_model():
    global model, classes, input_shape
    def clean_config_dict(d):
        """Recursively pops Keras 3 arguments that crash legacy loading paths"""
        if isinstance(d, dict):
            d.pop('quantization_config', None)
            d.pop('build_config', None)
            d.pop('shared_object_id', None)
            for k, v in list(d.items()):
                clean_config_dict(v)
        elif isinstance(d, list):
            for item in d:
                clean_config_dict(item)

    def translate_functional_to_model(d):
        """Recursively translates Functional class name to Model for model_from_json fallback"""
        if isinstance(d, dict):
            if d.get('class_name') == 'Functional':
                d['class_name'] = 'Model'
            for k, v in list(d.items()):
                translate_functional_to_model(v)
        elif isinstance(d, list):
            for item in d:
                translate_functional_to_model(item)

    try:
        print(f"[*] Memuat model Keras dari {MODEL_PATH}...")
        
        # Try normal load first
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("[+] Model berhasil dimuat secara langsung!")
        except Exception as e:
            print(f"[!] Gagal memuat langsung ({type(e).__name__}). Mencoba metode patch pembersihan konfigurasi...")
            
            # Read and patch JSON model config from HDF5 file
            with h5py.File(MODEL_PATH, 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config is None:
                    raise ValueError("Atribut 'model_config' tidak ditemukan di berkas H5. Pastikan ini adalah file model Keras lengkap, bukan hanya file weights.")
                
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                config_dict = json.loads(model_config)
                
                # Strip incompatible Keras 3 keys
                clean_config_dict(config_dict)
                
                # Try loading using Keras 3's legacy deserializer
                try:
                    from keras.src.legacy.saving import saving_utils
                    print("[i] Menggunakan deserializer legacy Keras untuk rekonstruksi model...")
                    model = saving_utils.model_from_config(config_dict)
                except Exception as legacy_err:
                    print(f"[!] Deserializer legacy gagal: {legacy_err}. Mencoba fallback ke model_from_json...")
                    
                    # Fallback path: translate Functional -> Model and use JSON loader
                    translate_functional_to_model(config_dict)
                    cleaned_config_json = json.dumps(config_dict)
                    model = tf.keras.models.model_from_json(cleaned_config_json)
                
            # Load weights into the patched structure
            model.load_weights(MODEL_PATH)
            print("[+] Model berhasil dimuat menggunakan metode patch pembersihan konfigurasi!")

        # Inspect model input shape
        model_input_shape = model.input_shape
        if isinstance(model_input_shape, list):
            model_input_shape = model_input_shape[0]
            
        if len(model_input_shape) == 4:
            input_shape = (model_input_shape[1], model_input_shape[2])
        print(f"[i] Input Shape model: {input_shape}")
        
        # Get output dimensions
        num_classes = model.output_shape[-1]
        print(f"[i] Jumlah kelas output model: {num_classes}")
        
        # Assign class metadata dynamically
        if num_classes in CLASS_MAPPINGS:
            classes = CLASS_MAPPINGS[num_classes]
            print(f"[i] Menggunakan mapping kelas otomatis untuk {num_classes} kelas.")
        else:
            # Fallback if class size is unknown
            classes = [{'name': f'Kategori {i}', 'desc': 'Kategori sampah terdeteksi.', 'recyclable': False, 'recyclable_text': 'Tidak Diketahui', 'recycle_tip': 'Kelola sesuai jenis bahan sampah.'} for i in range(num_classes)]
            print(f"[!] Jumlah kelas ({num_classes}) tidak cocok dengan mapping standar. Silakan sesuaikan di file app.py.")
            
        print("[i] Daftar Kelas yang digunakan:")
        for idx, cl in enumerate(classes):
            print(f"    - Indeks {idx}: {cl['name']} ({'Recyclable' if cl['recyclable'] else 'Non-Recyclable'})")
            
    except Exception as e:
        print("ERROR LOADING MODEL:")
        print(type(e).__name__)
        print(str(e))
        print(f"[-] Gagal memuat model: {e}")
        print("[!] Membuka aplikasi dalam mode simulasi dummy prediction.")

# Initial model loading
load_keras_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, classes, input_shape
    
    if 'image' not in request.files:
        return jsonify({'error': 'File gambar tidak ditemukan dalam request'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
    try:
        # Open the uploaded image file
        img = Image.open(file.stream)
        
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image according to model requirements
        img_resized = img.resize(input_shape)
        
        # Preprocessing image to NumPy array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalization - typically MobileNetV2 uses [-1, 1] range via preprocess_input
        # Here we perform common scaling: (x / 127.5) - 1.0
        # If your model was trained with [0, 1] normalization, change this line to: img_array = img_array / 255.0
        # For maximum safety, we do the MobileNetV2 preprocessing:
        img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension: shape becomes (1, height, width, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        if model is not None:
            # Predict using loaded model
            predictions = model.predict(img_array)
            # Find index of max confidence
            predicted_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_idx])
            
            # Form response
            if predicted_idx < len(classes):
                result = {
                    'class_index': predicted_idx,
                    'class_name': classes[predicted_idx]['name'],
                    'description': classes[predicted_idx]['desc'],
                    'recyclable': classes[predicted_idx]['recyclable'],
                    'recyclable_text': classes[predicted_idx]['recyclable_text'],
                    'recycle_tip': classes[predicted_idx]['recycle_tip'],
                    'confidence': confidence,
                    'all_predictions': [float(val) for val in predictions[0]]
                }
            else:
                result = {
                    'class_index': predicted_idx,
                    'class_name': f'Kategori {predicted_idx}',
                    'description': 'Kategori tidak dikenal.',
                    'recyclable': False,
                    'recyclable_text': 'Tidak Diketahui',
                    'recycle_tip': 'Pisahkan sampah dengan bijak.',
                    'confidence': confidence,
                    'all_predictions': [float(val) for val in predictions[0]]
                }
        else:
            # Dummy simulation when model is not available
            # Pick a random class and random confidence for testing
            import random
            sim_classes = CLASS_MAPPINGS[10] # Default to 10 classes simulation
            sim_idx = random.randint(0, len(sim_classes) - 1)
            result = {
                'class_index': sim_idx,
                'class_name': sim_classes[sim_idx]['name'],
                'description': sim_classes[sim_idx]['desc'],
                'recyclable': sim_classes[sim_idx]['recyclable'],
                'recyclable_text': sim_classes[sim_idx]['recyclable_text'],
                'recycle_tip': sim_classes[sim_idx]['recycle_tip'],
                'confidence': float(random.uniform(0.75, 0.98)),
                'simulated': True
            }
            
        return jsonify(result)
        
    except Exception as e:
        print(f"[-] Terjadi kesalahan saat prediksi: {e}")
        return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500

@app.route('/info', methods=['GET'])
def get_info():
    global classes
    return jsonify({
        'model_loaded': model is not None,
        'input_shape': list(input_shape),
        'classes': [c['name'] for c in classes]
    })

if __name__ == '__main__':
    # Run the server on port 7860
    app.run(host='0.0.0.0', port=7860, debug=True)