
import time

# Step 1: Load  data
lines = load_data("/content/training_set_chaos.txt")

# Step 2: Build vocabulary
vocab, inv_vocab = build_vocab(lines, vocab_size=1000)
vocab_size = len(vocab)


# Step 3: Tokenize the lines
tokenized = tokenize(lines, vocab)

# Step 4: Create batches
# We need sequences of at least length 2 for auto-regressive training
batches = create_batches([t for t in tokenized if len(t) > 1], batch_size=4)

# Step 5: Initialize Model, Loss, and Optimizer
model = BaselineTransformer(vocab_size=vocab_size, emb_dim=64, n_heads=4, n_layers=4)
losses = []
times=[]
# Ignore the padding index in the loss calculation for more accurate training
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
# Step 6: Training Loop
for epoch in range(100):
    epoch_loss = 0.0
    for batch in batches:
        optimizer.zero_grad()

        # --- Auto-regressive training ---
        # Input is all tokens except the last one
        inputs = batch[:, :-1]
        # Target is all tokens except the first one
        targets = batch[:, 1:]

        output = model(inputs)

        # Reshape for loss function
        # Output: (batch_size * seq_len-1, vocab_size)
        # Target: (batch_size * seq_len-1)
        loss = loss_fn(output.reshape(-1, vocab_size), targets.reshape(-1))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(batches)
    losses.append(avg_loss)
    times.append(time.time()-start_time)
    # Print loss every 10 epochs to avoid clutter
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
total_time = time.time() - start_time
print(f"Baseline Training time: {total_time:.2f} seconds")
# Step 7: Save losses (optional)
import json
with open("baseline_losses.json", "w") as f:
    json.dump(losses, f)

print("\n--- Training Complete ---")