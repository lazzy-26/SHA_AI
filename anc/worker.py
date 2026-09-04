def train_local_epoch(self):
    if self.train_loader is None:
        raise RuntimeError("Training loader has not been prepared.")
    
    self.model.train()
    total_loss = 0.0
    batches = 0
    start_time = time.time()
    
    # Gradient accumulation
    accumulation_steps = getattr(self, 'grad_accum_steps', 4)
    
    # Mixed precision
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    self.optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (noisy, clean) in enumerate(self.train_loader):
        if torch.cuda.is_available():
            noisy = noisy.cuda()
            clean = clean.cuda()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                enhanced = self.model(noisy)
                loss = self.criterion(enhanced, clean)
                loss = loss / accumulation_steps  # Normalize for accumulation
            scaler.scale(loss).backward()
            
            # Step every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            enhanced = self.model(noisy)
            loss = self.criterion(enhanced, clean)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        
        total_loss += float(loss.item() * accumulation_steps)
        batches += 1
    
    elapsed = time.time() - start_time
    average_loss = total_loss / max(batches, 1)
    
    return average_loss, elapsed, batches