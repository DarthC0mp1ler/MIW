clear close()
P = -2 :0.1: 2; T = P.^2 + 1*(rand(P)-0.5); 

//siec 
S1 = 100;
W1 = rand(S1, 1)- 0.5; B1 = rand(S1, 1)- 0.5;
W2 = rand(1, S1) -0,5; B2 = rand(1,1) -0.5;
lr = 0.001


for  epoka = 1 : 20
//odpowiedz sieci
A1 = tanh(W1*P + B1*ones(P));
A2 = W2*A1 + B2;

//propagacja wsteczna
E2 = T - A2;
E1 = W2'*E2;

dW2 = lr* E2 * A1';
dB2 = lr *E2 * ones(E2)';
dW1 = lr * (1 - A1.*A1) .* E1 * P';
dB1 = lr * (1 - A1.*A1) .* E1 * ones(P)';

W2 = W2 + dW2; B2 =  B2 + dB2;
W1 = W1 + dW1; B1 = B1 + dB1;


if modulo(epoka, 1)==0 then
clf();
plot(P,T, 'r*')
plot(P,A2)
sleep(500);
end

end
