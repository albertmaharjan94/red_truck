#define steer A2
int min_pot = 599;
int max_pot = 945;
int degree = 0;
void setup() {
  // put your setup code here, to run once:
  pinMode(steer, INPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  degree = map(analogRead(steer), min_pot, max_pot, 0 , 60);
  Serial.print(analogRead(steer));Serial.print("\t");Serial.println(degree);
  delay(500);
}
