#define power_pin 6
#define reverse_relay 2
#define steer_relay1 3
#define steer_relay2 4

// steering analog
#define steer A2


int voltage_bias = 0;
int stop_voltage = 0.83 * 51;

// initial forward to max threshold
int min_forward = 1.55 * 51;
int max_forward = 1.65 * 51;

// initial backward to max threshold
int min_backward = 1.68 * 51;
int max_backward = 2.30 * 51;

// initial voltage speed
int voltage = stop_voltage;

String state = "STOP";

int min_pot = 365 ;
int max_pot = 685 ;
int degree = 0;


int min_max_bias = 10;
bool goesLeft = false;
bool goesRight = false;


int delay_threshold = 2.5;
// function init
int getValue(String data, char separator, int index);


void setup() {
  //  power pin
  pinMode(power_pin, OUTPUT);
  // reverse for power
  pinMode(reverse_relay, OUTPUT);

  //  forward/reverse motor for steering
  pinMode(steer_relay1, OUTPUT);
  pinMode(steer_relay2, OUTPUT);

  //  analog pot
  pinMode(steer, INPUT);

  // Init to forward relay
  digitalWrite(reverse_relay, HIGH);
  Serial.begin(115200);

  //  Calibrate to middle degree 30
  int _tmp_steer = analogRead(steer);
  int _tmp_degree = map(_tmp_steer, min_pot, max_pot, 0 , 60);
  Serial.println("Calibrating to mid degree");
  //  if (_tmp_degree < 30){
  //    while(_tmp_degree == 30){
  //       _tmp_degree += 1;
  //       digitalWrite(steer_relay2, LOW);
  //       digitalWrite(steer_relay1, HIGH);
  //    }
  //  }else if(_tmp_degree > 30){
  //    while(_tmp_degree == 30){
  //       _tmp_degree -= 1;
  //       digitalWrite(steer_relay1, LOW);
  //       digitalWrite(steer_relay2, HIGH);
  //    }
  //  }
  degree = _tmp_degree;
  Serial.println("Back to mid degree");

}

void loop() {
  int steer_pot = analogRead(steer);
  degree = map(steer_pot, min_pot + min_max_bias, max_pot - min_max_bias, 0 , 60);

  //  steering logic
  if (goesRight == true &&  steer_pot >= min_pot + min_max_bias) {
    digitalWrite(steer_relay1, LOW);
    digitalWrite(steer_relay2, HIGH);
  } else if (goesLeft == true && steer_pot <= max_pot - min_max_bias) {
    digitalWrite(steer_relay2, LOW);
    digitalWrite(steer_relay1, HIGH);
  } else {
    goesRight = false;
    goesLeft = false;
    digitalWrite(steer_relay2, LOW);
    digitalWrite(steer_relay1, LOW);
  }

  Serial.flush();
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    //String data = "90#20#0\n";

    int _speed = getValue(data, '#', 0);
    int _steer = getValue(data, '#', 1);
    int _brake = getValue(data, '#', 2);

    _speed = _speed != -1 ? _speed : stop_voltage;
    _steer = _steer != -1 ? _steer : 230;
    _brake = _brake != -1 ? _brake : 0;
    Serial.print(_speed); Serial.print(" "); Serial.print(_steer); Serial.print(" "); Serial.print(_brake); Serial.println(" ");

    if (_speed >= -100 && _speed <= 100) {
      if (_speed == 0 ) {
        if (state == "FORWARD" || state == "BACKWARD") {
          // decrease speed linearly
          for (int i = voltage; i >= stop_voltage; i = i - 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }
        }
        state = "STOP";
      }

      if (_speed > 0) {
        if (state == "BACKWARD") {
          // decrease speed linearly
          for (int i = voltage; i >= stop_voltage; i = i - 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }
        }

        // switch relay for forward
        digitalWrite(reverse_relay, HIGH);
        if (state == "BACKWARD" || state == "STOP" ) {
          // increase speed linearly
          for (int i = min_forward; i <= max_forward; i = i + 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }
          for (int i = min_forward; i <= max_forward; i = i + 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }
        }
        digitalWrite(reverse_relay, HIGH);
        state = "FORWARD";
      }
      else if (_speed < 0) {
        if (state == "FORWARD") {
          // decrease speed linearly
          for (int i = voltage; i >= stop_voltage; i = i - 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }
        }
        // switch relay for backward
        digitalWrite(reverse_relay, LOW);
        if (state == "FORWARD" || state == "STOP") {
          // increase speed linearly
          for (int i = min_backward; i <= max_backward; i = i + 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }

          for (int i = min_backward; i <= max_backward; i = i + 2) {
            voltage = i;
            analogWrite(power_pin, voltage);
            Serial.print(state); Serial.print("  ");
            Serial.println(voltage);
            delay(delay_threshold);
          }
        }
        state = "BACKWARD";
      }

      if (_steer >= 200 && _speed <= 260) {
        if (_steer > 230) {
          degree = map(steer_pot, min_pot + min_max_bias, max_pot - min_max_bias, 0 , 60);
          if (_steer != degree) {
            goesLeft = true;
            goesRight = false;
          }
        }
        else if (_steer < 230) {
          degree = map(steer_pot, min_pot + min_max_bias, max_pot - min_max_bias, 0 , 60);
          if (_steer != degree) {
            goesRight = true;
            goesLeft = false;
          }

        } else if (_steer == 230) {
          goesRight = false;
          goesLeft = false;
          digitalWrite(steer_relay2, LOW);
          digitalWrite(steer_relay1, LOW);
        }
      }
    }
    analogWrite(power_pin, voltage);
  }
  delay(delay_threshold);
}


int getValue(String data, char separator, int index)
{
  int found = 0;
  int strIndex[] = { 0, -1 };
  int maxIndex = data.length() - 1;

  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == separator || i == maxIndex) {
      found++;
      strIndex[0] = strIndex[1] + 1;
      strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }
  return found > index ? data.substring(strIndex[0], strIndex[1]).toInt() : -1;
}
