import json
import h5py
import tensorflow as tf

def clean_config_dict(d):
    """Recursively pops Keras 3 arguments that crash Keras 2"""
    if isinstance(d, dict):
        d.pop('quantization_config', None)
        d.pop('build_config', None)
        d.pop('shared_object_id', None)
        for k, v in list(d.items()):
            clean_config_dict(v)
    elif isinstance(d, list):
        for item in d:
            clean_config_dict(item)

def inspect_h5():
    model_path = 'garbage_classifier_mobilenetv2.h5'
    try:
        print(f"Loading Keras model from {model_path}...")
        
        # Try normal load first
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully directly!")
        except Exception as e:
            print(f"Direct load failed ({type(e).__name__}). Trying configuration patching method...")
            
            # Read and patch JSON model config from HDF5 file
            with h5py.File(model_path, 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config is None:
                    raise ValueError("Atribut 'model_config' tidak ditemukan di berkas H5. Pastikan ini adalah file model Keras lengkap, bukan hanya file weights.")
                
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                config_dict = json.loads(model_config)
                
                # Strip incompatible Keras 3 keys
                clean_config_dict(config_dict)
                
                cleaned_config_json = json.dumps(config_dict)
                
                # Reconstruct architecture from patched JSON
                model = tf.keras.models.model_from_json(cleaned_config_json)
                
            # Load weights into the patched structure
            model.load_weights(model_path)
            print("Model loaded successfully using patched loader!")

        print("\n=== MODEL SUMMARY ===")
        model.summary()
        
        # Get input and output shapes
        print("\nInput shape:", model.input_shape)
        print("Output shape:", model.output_shape)
        
    except Exception as e:
        print("\nError reading model:", e)

if __name__ == "__main__":
    inspect_h5()
