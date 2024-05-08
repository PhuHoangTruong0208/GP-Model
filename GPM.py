import torch
from torch import nn, optim

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# biến đầu vào thành các token
class Tokenizer:
    def __init__(self, max_sequence_length, pad="<pad>", end="<end>"):
        self.max_sequence_length = max_sequence_length
        self.index = [pad, end]
        self.pad = pad
        self.end = end
    
    def tokenize(self, x, y, sentence, to_index=False, index_to=False):
        for lines in x + y:
            for word in lines.split():
                if word not in self.index:
                    self.index.append(word)
        
        self.t_idx = {v:k for k, v in enumerate(self.index)}
        self.idx_t = {k:v for k, v in enumerate(self.index)}

        tokenize = []
        words = []
        
        if to_index:
            for word in sentence.split():
                tokenize.append(self.t_idx[word])
            for _ in range(len(tokenize), self.max_sequence_length):
                tokenize.append(self.t_idx[self.pad])
        
        if index_to:
            print(sentence)
            for num in sentence:
                words.append(self.idx_t[num])

        if to_index and index_to:
            return tokenize, words 
        if to_index:
            return tokenize
        if index_to:
            return words

# lấy vector có xác suất cao nhất
def get_most_vector(tensor):
    num= torch.tensor(-1)
    vector_most=None
    for vector in tensor:
        argmax = torch.argmax(vector)
        if argmax > num:
            num = argmax
            vector_most = vector
    return vector_most

# chuyển vector tokenize thành vector ngữ cảnh
class ToVectorContex(nn.Module):
    def __init__(self, context_units, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, context_units).to(device)
        self.context_layer = nn.Linear(context_units, context_units).to(device)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.context_layer(x)
        return x

# Dự đoán từng từ tương ứng từ số lượng mạng tương ứng
class DenseGenerativeWord(nn.Module):
    def __init__(self, max_sequence_length, context_units, vocab_size):
        super().__init__()
        self.dense_vector_context = [nn.Linear(context_units, context_units).to(device) for _ in range(max_sequence_length)]
        self.dense_vector_words = [nn.Linear(context_units, vocab_size).to(device) for _ in range(max_sequence_length)]
        self.context_units = context_units
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x, context_input):
        context_list, word_predict_list = [], []
        for layer in self.dense_vector_context:
            if not isinstance(context_input, list):
                x = layer(x + context_input)
            else:
                x = layer(x)
            context_list.append(x)
        
        for i in range(len(context_list)):
            context = self.dense_vector_words[i](context_list[i])
            word_predict_list.append(self.softmax(context))
        return word_predict_list, context_list
    

class GPMLayer(nn.Module): # GPM -> Generative Private Model
    def __init__(self, max_sequence_length, context_units, vocab_size):
        super().__init__()
        self.to_context_layer = ToVectorContex(context_units, vocab_size)
        self.dense_generative_word = DenseGenerativeWord(max_sequence_length, context_units, vocab_size)

    def forward(self, x, context_input):
        x = self.to_context_layer(x)
        x, context_output = self.dense_generative_word(x, context_input)
        return x, context_output


# dự đoán từ
class GenerativePrivateModel(nn.Module):
    def __init__(self, max_sequence_length, context_units, vocab_size):
        super().__init__()
        self.model = GPMLayer(max_sequence_length, context_units, vocab_size)
        # lưu trữ ngữ cảnh
        self.context = None
        
    def forward(self, x):
        predict_tensor = []
        for f in x:
            # tạo ngữ cảnh đầu vào
            context_input = []
            if self.context != None:
                for context_l in self.context:
                    for context_d in context_l:
                        context_input.append(context_d)
                context_input = [torch.stack(context_input)]
            # nhận đầu ra và ngữ cảnh đầu ra
            result, context_output = self.model(f, context_input)
            self.context = [context.detach() for context in context_output]
            # dự đoán
            predict = []
            for sample_prediction in result:
                predict.append(get_most_vector(sample_prediction))
            predict = torch.stack(predict)
            predict_tensor.append(predict)
        predict_tensor = torch.stack(predict_tensor)
        return predict_tensor
    
# dự đoán từ
def predict(model, inp):
    inps = []
    for w in inp.split():
        inps.append(tokenizer.t_idx[w])
    inps = torch.tensor([inps])
    result = model(inps)

    sentence = ""
    for tensor in result[0]:
        arg = int(torch.argmax(tensor))
        sentence += tokenizer.idx_t[arg]+" "
        if tokenizer.idx_t[arg] in tokenizer.end:
            break
    return sentence

            
x = ["xin chào", "bạn tên là gì", "bạn có khỏe không", "mấy giờ rồi", "khi nào bạn đi", "bạn đang làm gì vậy",
     "tạm biệt nha", "chào buổi sáng", "hôm nay bạn định làm gì", "chà hôm nay không làm gì à",
     "à haha tôi quên"]

y = ["chào <end>", "tôi tên là bot <end>", "tôi khỏe <end>", "12h rồi ạ <end>", "một chút nữa <end>", "tôi đang rãnh không làm gì <end>",
     "tạm biệt hẹn gặp lại sau <end>", "xin chào chào buổi sáng <end>", "tôi định nấu ăn sau đó đi ngủ <end>",
     "hôm nay là chủ nhật đấy <end>", "không sao nhưng quên ngày chủ nhật thì lạ thật <end>"]

tokenizer = Tokenizer(max_sequence_length=25)
x_tensor = []
y_tensor = []

for seq in x:
    x_tensor.append(tokenizer.tokenize(x, y, sentence=seq, to_index=True))
for seq in y:
    y_tensor.append(tokenizer.tokenize(x, y, sentence=seq, to_index=True))

x_tensor = torch.tensor(x_tensor)
y_tensor = torch.tensor(y_tensor)

X_train = x_tensor
y_train = y_tensor

model = GenerativePrivateModel(max_sequence_length=25, context_units=512, vocab_size=len(tokenizer.index))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 300
for epoch in range(num_epochs):
    print("Epochs ", epoch)
    model.train()
    running_loss = 0.0

    inputs, labels = X_train, y_train
    optimizer.zero_grad()

    outputs = model(inputs)
    # print("đầu ra: ", outputs.view(-1, len(tokenizer.index)).size())
    # print("nhãn: ", labels.view(-1).size())
    loss = criterion(outputs.view(-1, len(tokenizer.index)), labels.view(-1))
        
    loss.backward(retain_graph=True)
    optimizer.step()

    running_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print("X: chào bạn tên gì")
        print("Y Predict: ", predict(model, inp=" chào bạn tên gì"))
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{epoch+1}/{len(X_train)}], Loss: {running_loss/10}')
        running_loss = 0.0

print('Training Finished')


while True:
    inp = input("Bạn: ")
    predicts = predict(model=model, inp=inp)
    print(f"bot: {predicts}")
