#define steer_relay1 3
#define steer_relay2 4
#define steer A2
// 379 712
int min_pot = 385 ;
int max_pot = 690 ;
int degree = 0;

int min_max_bias = 10;
bool goesLeft = false;
bool goesRight = false;
void setup() {
  // put your setup code here, to run once:
  pinMode(steer_relay1, OUTPUT);
  pinMode(steer_relay2, OUTPUT);
  pinMode(steer, INPUT);
  digitalWrite(steer_relay1, HIGH);
  digitalWrite(steer_relay2, HIGH);
  Serial.begin(9800);
}

void loop() {
  // put your main code here, to run repeatedly:
  int steer_pot = analogRead(steer);
  
  Serial.print(degree);
  Serial.print("  ");
  Serial.print(goesRight);
  Serial.print("  ");
  Serial.print(goesLeft);
  Serial.print("  ");

  
  
  Serial.println(steer_pot);
  
  degree = map(steer_pot, min_pot+min_max_bias, max_pot-min_max_bias, 0 , 60);
  if (goesRight == true &&  steer_pot >= min_pot + min_max_bias) {
    digitalWrite(steer_relay1, LOW);
    digitalWrite(steer_relay2, HIGH);
  } else if (goesLeft == true && steer_pot <= max_pot - min_max_bias) {
    digitalWrite(steer_relay2, LOW);
    digitalWrite(steer_relay1, HIGH);
  } else {
    goesRight = false;
    goesLeft = false;
    Serial.println("stop");
    digitalWrite(steer_relay2, LOW);
    digitalWrite(steer_relay1, LOW);
  }

  if (Serial.available()) {
    char data = Serial.read();
    if (data == 'w') {
      goesLeft = true;
      goesRight = false;
    }
    else if (data == 'x') {
      goesRight = true;
      goesLeft = false;
    } else if (data == 's') {
      goesRight = false;
      goesLeft = false;
      digitalWrite(steer_relay2, LOW);
      digitalWrite(steer_relay1, LOW);
    }


    Serial.print(data); Serial.print("  ");

  }
  //  }

}
