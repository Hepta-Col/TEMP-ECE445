<template>
  <div id="Data">
   {{ "Lastest update on" }} {{ weather_data.day }} {{ "at" }} {{ weather_data.time }} 
   <br>
   <br>
   {{ "Temperature:" }} {{ weather_data.temp }} {{ "\u2103" }}
   <br>
   {{ "Humidity: " }} {{ weather_data.humi }} {{ "%" }}
   <br>
   {{ "Pressure: " }} {{ weather_data.pres }} {{ "hPa" }}
   <br>
   {{ "Wind: " }} {{ weather_data.wind }} {{ "m/s" }}
   <br>
   {{ "UV Index: " }} {{ weather_data.uv }}
   <br>
   {{ "Air Quality:" }} {{ "PM1.0: " }} {{ weather_data.pm10 }} {{ "ug/m^3" }} | {{ "PM2.5: " }} {{ weather_data.pm25 }} {{ "ug/m^3" }} | {{ "PM10: " }} {{ weather_data.pm100 }} {{ "ug/m^3" }}
   <br>
   <div v-if="weather_data.rain > 0">{{ "Rainy day" }}</div>
   <div v-else>{{ "Clear! No rain now!" }}</div>
   <ul>
      <span v-for="(value, key) in hourly_pred">
        {{ "At hour" }} {{ key }} {{ ":" }} {{ value }} {{ "\n" }}
      </span>
   </ul>
   <li v-for="(value, key) in daily_pred">{{ "On" }} {{ key }} {{ ":" }} {{ value[0] }} - {{ value[1] }}</li>
   <br>
  </div>
</template>

<script>
export default {
  name: 'Data',
  data () {
    return {
      page_info: 'This is Data page',
      weather_data: "",
      hourly_pred: "",
      daily_pred: ""
    }
  },
  created() {
    this.$axios.get("http://localhost:8000/data/")
      .then(response =>{
        this.weather_data = response.data
      })
      .catch((error) => {
        console.log(error);
      });
      this.$axios.get("http://localhost:8000/data/hourlypredict/")
      .then(res =>{
        this.hourly_pred = res.data
      })
      .catch((error) => {
        console.log(error);
      });
      this.$axios.get("http://localhost:8000/data/dailypredict/")
      .then(res =>{
        this.daily_pred = res.data
      })
      .catch((error) => {
        console.log(error);
      });
  },
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
