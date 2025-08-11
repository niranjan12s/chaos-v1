# --- Chaos Data Setup ---
lines = load_data("/content/training_set_chaos.txt")

# Build the vocab (same function as baseline)
vocab, inv_vocab = build_vocab(lines, vocab_size=1000)
vocab_size = len(vocab)

# Tokenize (again, same as baseline)
tokenized = tokenize(lines, vocab)

# Filter short ones, pad, and batch
batches = create_batches([t for t in tokenized if len(t) > 1], batch_size=4)


import time
vocab_size=1000
model = ChaosTransformer(
    vocab_size=vocab_size,
    num_dims=16,
    dim_vec_size=16,
    emb_dim=64,
    n_heads=4,
    n_layers=4,
    ff_hidden=32
)

losses = []
times=[]
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dim_grad_accum = torch.zeros(16)  # For tracking
start_time = time.time()
for epoch in range(100):
    epoch_loss = 0.0
    for batch in batches:
        optimizer.zero_grad()
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        output, dim_weights = model(inputs)
        loss = loss_fn(output.view(-1, vocab_size), targets.reshape(-1))

        loss.backward()

        grads = model.dim_emb.dimension_vectors.grad  # [num_dims, dim_vec_size]
        dim_grad_accum += grads.abs().mean(dim=1).cpu()

        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(batches)
    losses.append(avg_loss)
    times.append(time.time()-start_time)

    if (epoch + 1) % 10 == 0:
        print(f"[CHAOS] Epoch {epoch+1}, Loss: {avg_loss:.4f}")
total_time = time.time() - start_time
print(f"Chaos Training time: {total_time:.2f} seconds")

# Save
with open("chaos_losses.json", "w") as f:
    json.dump(losses, f)

print("--- CHAOS Model Training Complete ---")
