def train_model(model, X, y, epochs=20, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
