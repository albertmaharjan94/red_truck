int brake = 5;
void setup() {
  // put your setup code here, to run once:
  pinMode(brake, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(brake, HIGH);
  delay(4000);
  digitalWrite(brake, LOW);
  delay(4000);
}
