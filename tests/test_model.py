from ml_model.train import train_model

def test_train_model():
    # Train the model
    model = train_model()
    assert model is not None  # Ensure the model is trained

def test_prediction():
    # Test if prediction works with the model
    from ml_model.predict import predict
    sample_input = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]  # Example input for California dataset
    prediction = predict(sample_input)
    assert len(prediction) == 1  # Ensure a single prediction is returned

