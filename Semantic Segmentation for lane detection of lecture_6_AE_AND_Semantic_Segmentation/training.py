def fit(epochs,model,train_dl,cost_fn,optimizer,device="cuda"):
	train_cost_per_epoch = []
	model.to(device)

	for epoch in range(1,epochs+1):
		model.train()
		train_cost=0
		for _, (x,y) in enumerate(train_dl):
			optimizer.zero_grad()
			x = x.to(device)
			y = y.to(device)
			pred = model(x)
			error = cost_fn(pred,y)
			error.backward()
			optimizer.step()
			train_cost = error.item() +train_cost


		train_cost_per_epoch.append(train_cost)
		torch.save(model.state_dict(),r"C:\Users\monsi\Desktop\project\lane_detection\inchalah.pt")
		if epoch % 10 == 0:
			print(f'******** model saved at epoch ********: {epoch}')

	return train_cost_per_epoch
