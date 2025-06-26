"""
Setup script for Currency Detection System
Initializes the system and creates sample data
"""
import os
import logging
from utils.data_loader import CurrencyDataLoader, create_sample_dataset
from currency_detector import CurrencyDetectionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_system():
    """Set up the currency detection system"""
    
    print("🚀 Setting up Advanced Currency Detection System...")
    print("="*60)
    
    # Initialize components
    logger.info("Initializing system components...")
    
    try:
        # Create data loader
        data_loader = CurrencyDataLoader()
        
        # Set up directory structure
        logger.info("Setting up directory structure...")
        data_loader.download_sample_datasets()
        
        # Create sample dataset for demonstration
        logger.info("Creating sample dataset...")
        sample_data = create_sample_dataset()
        
        # Initialize detection system
        logger.info("Initializing detection system...")
        detector = CurrencyDetectionSystem()
        
        # Print system information
        detector.print_system_info()
        
        print("\n✅ Setup completed successfully!")
        print("\n📝 Next Steps:")
        print("1. Add your currency images to the data folders:")
        print("   - data/USD/real/ (for real USD notes)")
        print("   - data/USD/fake/ (for fake USD notes)")
        print("   - data/SAR/real/ (for real SAR notes)")
        print("   - data/SAR/fake/ (for fake SAR notes)")
        print("\n2. Run the web application:")
        print("   streamlit run web_app.py")
        print("\n3. Or use the Python API:")
        print("   from currency_detector import CurrencyDetectionSystem")
        
        print(f"\n📊 Sample Data Created:")
        for currency, (images, labels) in sample_data.items():
            real_count = sum(labels)
            fake_count = len(labels) - real_count
            print(f"   {currency}: {len(images)} images ({real_count} real, {fake_count} fake)")
        
        print("\n🔧 Configuration:")
        print("   Edit config.py to customize system parameters")
        
        print("\n📚 Documentation:")
        print("   See README.md for detailed usage instructions")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_system()
    if success:
        print("\n🎉 Currency Detection System is ready to use!")
    else:
        print("\n💔 Setup encountered errors. Please check the logs.")