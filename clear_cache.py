# clear_cache.py
import os
import shutil

def clear_streamlit_cache():
    # Find and clear Streamlit cache directory
    streamlit_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
    
    if os.path.exists(streamlit_dir):
        print(f"Clearing Streamlit cache at: {streamlit_dir}")
        try:
            shutil.rmtree(streamlit_dir)
            print("✅ Streamlit cache cleared successfully!")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    else:
        print("ℹ️ No Streamlit cache directory found")

if __name__ == "__main__":
    clear_streamlit_cache()