import os
import joblib
from train_models import models, preprocessor

# Create models directory if it doesn't exist
os.makedirs('../models/saved_models', exist_ok=True)

# Save each trained model
for name, model in models.items():
    # Create a filename for the model
    filename = f'../models/saved_models/{name.lower().replace(" ", "_")}_model.joblib'
    
    # Save the model
    joblib.dump(model, filename)
    print(f"Saved {name} model to {filename}")

# Save the preprocessor
joblib.dump(preprocessor, '../models/saved_models/preprocessor.joblib')
print("Saved preprocessor to ../models/saved_models/preprocessor.joblib")

print("\nAll models have been successfully extracted and saved!") 