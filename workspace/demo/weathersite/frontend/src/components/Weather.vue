<template>
  <div id="Weather">
   <!-- {{ msg_info }} -->

   <el-container>
  <el-header height = "40px">Welcome to our weather station</el-header>
  <el-container>
    <el-main>
      <div class = "content-box">
        <br>
        <h2 style="font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif, 'Arial Narrow', Arial, sans-serif ;margin-bottom: 0em; font-weight: bold;"> {{ "Haining Campus" }}</h2>
        <h1 style="color:Tomato; font-size:xxx-large; margin-top: 0.2em; margin-bottom: 0.3em; font-weight: bold;"> {{ weather_data.temp }} {{ "\u2103" }} <a v-if="weather_data.rain"> <i class="icon iconfont icon-tianqi-yutian" style="font-size: 40px;"></i> </a> </h1>
        <h6 style="margin-top: 0em;"> {{ "Last update on" }}  {{ weather_data.day }} {{ "at" }} {{ weather_data.time }}</h6>
      </div>

    </el-main>
    <el-aside width=50%>
      <el-row></el-row>
      <el-row></el-row>
      <el-row>
          <el-col :span="9" :offset="0">
            <el-container>
              <!-- <el-header height = "40px" style="background-color: red; text-align:left; ">Humidity</el-header> -->
              
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-shidu"></i> {{ " Humidity" }} <br/> </div>
                <div class = "br2" style="font-size:x-large; font-weight: 500;"> <br/> {{ weather_data.humi }} {{ "%" }} <br/> </div>
              </el-main>
            </el-container>
            <div class = "br3"><br/></div>
            <el-container>
              <el-tooltip class="item" effect="dark" :content="'Level ' + weather_data.wind_level + ': ' +weather_data.wind_msg" placement="bottom">
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-dafeng"></i> {{ " Wind" }} <br/> </div>
                <div class = "br2" style="font-size:x-large; font-weight: 500;"> <br/> <a v-if="[6,7,8,9,10,11].includes(weather_data.wind_level)"><i class="icon iconfont icon-alarm" style="font-size: 25px; color: brown;"></i></a> {{ weather_data.wind }} {{ "m/s" }} <br/> </div>
      
              </el-main>
              </el-tooltip>
            </el-container>
          </el-col>
          <el-col :span="10" :offset="2">
            <el-container>
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-daqiyali"></i> {{ " Pressure" }} <br/> </div>
                <div class = "br2"><br/></div>
                <div id="myChart" :style="{width: '180px', height: '126px'}"></div>
                <!-- <div class = "br3"><br/></div> -->
              </el-main>
              
            </el-container>

          </el-col>
      </el-row>
    </el-aside>
  </el-container>
  <el-footer  height = "200px">
    <el-row>
          <el-col :span="8" :offset="0">
            <el-container>
              <!-- <el-header height = "40px" style="background-color: red; text-align:left; ">Humidity</el-header> -->
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br4" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-ziwaixian"></i> {{ " UV Index" }} <br/> </div>
                <div class = "br1" style="font-size:x-large; font-weight: 600;"> {{ weather_data.uv }} <span v-if="[11].includes(weather_data.uv)" style="font-weight: bold; margin-left: -8px;">+</span>
                  <br/> <span v-if="[0,1,2].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;"> Low </span> </span> 
                  <span v-if="[3,4,5].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">Moderate</span></span>
                  <span v-if="[6,7].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">High</span></span> 
                  <span v-if="[8,9,10].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">Very High</span></span>
                  <span v-if="[11].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">Extreme</span></span>
                  <br/>
                </div>
                <div class = "br4" style="font-weight:200;"> 
                  <span v-if="[0,1,2].includes(weather_data.uv)"><span><i class="icon iconfont icon-mojing-" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Safely enjoy being outside. </span></span> 
                  <span v-if="[3,4,5].includes(weather_data.uv)"><span><i class="icon iconfont icon-huabanfuben" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Apply broad spectrum sunscreen. </span></span>
                  <span v-if="[6,7].includes(weather_data.uv)"><span><i class="icon iconfont icon-yusan" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Reduce time in the sun.</span></span> 
                  <span v-if="[8,9,10].includes(weather_data.uv)"><span><i class="icon iconfont icon-fangshaiyi" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Take extra precautions for skin.</span></span>
                  <span v-if="[11].includes(weather_data.uv)"><span><i class="icon iconfont icon-shineimdpi" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Avoid being outside during midday.</span></span>
                  <br/>
                </div>
                <!-- <el-progress type="line" :percentage="uv_percentage" :show-text="false"></el-progress>  -->
                <div class="cp-progress-main">
                  <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                    <div class="cp-progress-bar" :style="{ width: getPercentage+'%' ,background:getGradient,height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div></div>
                </div>
              </el-main>
            </el-container>
            <div class = "br3"><br/></div>
          </el-col>
          <el-col :span="14" :offset="1">
            <el-container>
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;">{{ " Air Quality" }} <br/> </div>
                <div class = "br2"><br/></div>
                <div id="myChart" :style="{width: '180px', height: '126px'}"></div>
                <!-- <div class = "br3"><br/></div> -->
              </el-main>
              
            </el-container>

          </el-col>
      </el-row>
  </el-footer>
</el-container>
 
  <br>

   <el-row :gutter="20">
   <el-col :span="16"><div class="grid-content bg-purple"> {{ "Temperature:" }} {{ weather_data.temp }} {{ "\u2103" }}</div></el-col>
   <el-col :span="8"><div class="grid-content bg-purple"> {{ "Humidity: " }} {{ weather_data.humi }} {{ "%" }}</div></el-col>
   </el-row>

   <el-row :gutter="20">
   <el-col :span="6"><div class="grid-content bg-purple">{{ "Temperature:" }} {{ weather_data.temp }} {{ "\u2103" }}</div></el-col>
   <el-col :span="6"><div class="grid-content bg-purple">{{ "Humidity: " }} {{ weather_data.humi }} {{ "%" }}</div></el-col>
   <el-col :span="6"><div class="grid-content bg-purple">{{ "Pressure: " }} {{ weather_data.pres }} {{ "hPa" }}</div></el-col>
   <el-col :span="6"><div class="grid-content bg-purple"> {{ "Wind: " }} {{ weather_data.wind }} {{ "m/s" }}</div></el-col>
   </el-row>

  </div>

</template>

<script>
var echarts = require('echarts');
export default {
  name: 'Weather',
  data () {
    return {
      msg_info: 'Play with elementUI',
      weather_data: "",
      hourly_pred: "",
      daily_pred: "",
      draw_pressure: "",
      uv_percentage: "",
      // pressure_percentage: "",
      colors: [
          {color: '#f56c6c', percentage: 20},
          {color: '#e6a23c', percentage: 40},
          {color: '#5cb87a', percentage: 60},
          {color: '#1989fa', percentage: 80},
          {color: '#6f7ad3', percentage: 100}
        ],
    }
  },
  computed: {
    // 通过比例 获取 百分比
    getPercentage () {
      if (this.uv_percentage < 0) {
        return 0
      } else if (this.uv_percentage > 1) {
        return 100
      } else {
        // console.log(this.percentage)
        return parseInt((this.uv_percentage + 0.000006) * 100)
      }
    },
    // 获取 进度条颜色对象
    getGradient () {
      let linecolor = this.getColorItem(this.uv_percentage)
      if (linecolor) {
        return 'linear-gradient(90deg,' + linecolor.s + ',' + linecolor.e + ')'
      } else {
        return ''
      }
    }
  },
  mounted(){
    this.drawline();
  },
  created() {
    this.$axios.get("http://localhost:8000/data/")
      .then(response =>{
        this.weather_data = response.data
        this.draw_pressure = this.weather_data.pres
        this.uv_percentage = (this.weather_data.uv +1) /12
        this.drawline()
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
  methods: {
    drawline(){
      let myChart = this.$echarts.init(document.getElementById('myChart'))
      myChart.setOption({
        // tooltip: {
        // formatter: '{a} <br/>{b} : {c}%'
        // },
      series: [{
            name: 'pressure',
            type: 'gauge',
            max: 1060,
            min: 860,
            startAngle: 225,
            endAngle: -45,
            progress: {
              show: false,
            },
            pointer: { // 指针样式
                width: 3,
                length: '60%',
                shadowBlur: 5,
                show: true,
                // offsetCenter: ['-20%', 0],
                itemStyle: {
                    color: '#99a9bf',
                    padding: [50,0,0,0]
                }
            },
            detail: {
                valueAnimation: true,
                formatter: ['{value}', '{a|hPa}'].join('\n'),
                fontSize: 20,
                width: '100%',
                height: '30%',
                padding: [-80, 0, 0, 0],
                rich: {
                    a: {
                        color: '#68A54A',
                        fontSize: 16,
                        padding: [3, 0, 10, 0],
                    }
                }
            },
            data: [{
                value: this.draw_pressure,
            }],
            axisLine: {
                lineStyle: {
                    color: [ //数组第一个属性是颜色所占line百分比
                        [0.3, "#49afff"],
                        [0.7, "#68A54A"],
                        [1, "#f56c6c"],

                    ],
                    width: 10
                }
            },
            splitLine: {
                // length: 5,
                // lineStyle: {
                //     width: 3
                // }
                show: false
            },
            axisLabel: {
                distance: -40,
                show: true,
                formatter: function(value) {
                    if (value === 860) {
                        return "Low"
                    }
                    if(value == 1060){
                        return "High"
                    }
                    return ''
                },
                //   padding: '8 0 0 0'
                lineHeight: -60,
                fontSize: 15,
                fontWeight: "bold"
            },
            axisTick: {
                show: false
            },
            animation: true,
            animationEasingUpdate: "quadraticIn"
        },
      ]
      });
    },
    getColorItem (p) {
      let mp = this.getPercentage
      for (let sub of this.linearColors) {
        if (!sub.ef && mp <= sub.v) {
          return sub
        } else if (sub.ef && mp < sub.v) {
          return sub
        }
      }
      return null
    }

  },
  props: {
    // 设置 进度条的 弧度
    bRadius: {
      type: Number,
      default: 4
    },
    textInside: {
      type: Number,
      default: 100
    },
    // 进度条的高度 就是粗细度
    strokeWidth: {
      type: Number,
      default: 8
    },
    // 进度条 的百分比 [0-1] 的小数
    percentage: {
      type: Number,
      default: 0
    },
    // 进度条 每个阶段的 颜色组
    linearColors: {
      type: Array,
      default: function () {
        return [{ v: 25, s: '#00ff00', e: '#ffff00' }, { v: 50, s: '#00ff00', e: '#FFA500' }, { v: 66.67, s: '#00ff00', e: '#FFA500', ef: true }, { v: 91.66, s: '#00ff00', e: '#ff6f00' }, { v: 100, s: '#00ff00', e: '#FF0000' }]
      }
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
body {
  max-width: 100%;
  margin: 0 auto;
}
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
  color: black;
}
.icon, .iconfont {
  font-family:"iconfont" !important;
  font-size:20px;
  font-style:normal;
  -webkit-font-smoothing: antialiased;
  -webkit-text-stroke-width: 0.2px;
  -moz-osx-font-smoothing: grayscale;
}
span {
  font-weight: 200;
}

  .el-row {
    margin-bottom: 10px;
    &:last-child {
      margin-bottom: 0;
    }
  }
  .el-col {
    border-radius: 4px;
  }

  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }
  .row-bg {
    padding: 10px 0;
    background-color: #f9fafc;
  }

  .br1{ line-height:30px}
  .br2{ line-height:15px}
  .br3{ line-height:20px}
  .br4{ line-height:40px}

  .el-header {
    background-color: #F2F6FC;
    color: #333;
    text-align: center;
    font-family:Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
    line-height: 40px;
  }

  .el-footer {
    background-color: #ffffff;
    color: #333;
    text-align: center;
    /* font-family:Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif; */
    /* line-height: 40px; */
  }
  
  .el-aside {
    background-color:white;
     /* #F2F6FC; */
    color: #333;
    text-align: center;
    line-height: 40px;
  }
  
  .el-main {
    background-color: #ffffff;
    color: #333;
    text-align: center;
    /* line-height: 30px; */
  }
  
  body > .el-container {
    margin-bottom: 40px;
  }

  .el-container:nth-child(5) .el-aside,
  .el-container:nth-child(6) .el-aside {
    line-height: 260px;
  }
  
  .el-container:nth-child(7) .el-aside {
    line-height: 320px;
  }

  .h1 {
  display: block;
  font-size: 2em;
  margin-top: 0.67em;
  margin-bottom: 0.67em;
  margin-left: 0;
  margin-right: 0;
  font-weight: bold;
  } 

  .text-wrapper {
  white-space: pre-wrap;
  }

  .p1 {
    word-break: break-all;
    width: 150px;
  }

  .el-progress{
  width:100%;		
  }
  .el-progress__text{
    color: #fff;
    font-size: 14px;
  }

  .progressbar >>> .el-progress-bar .el-progress-bar__outer {
    background-color: transparent;
  }

  .progressbar >>> .el-progress-bar .el-progress-bar__outer .el-progress-bar__inner {
    background-image: linear-gradient(to right, #00ff00 , #ffff00, #FFA500, #FF0000, #E6E6FA);
  }

  .progress{
  width: 500px;
  height: 20px;
  padding-left: 10px;
}

.progress ::v-deep .el-progress__text{
  color: #fff;
  font-size: 14px;
}
.progress ::v-deep .el-progress-bar__outer{
  height: 12px!important;
  border: 1px solid #78335f;
  background-color:transparent;
}

.progress ::v-deep .el-progress-bar__inner{
background-color: unset;
  background-image: linear-gradient(to right, #00ff00 , #ffff00, #FFA500, #FF0000, #E6E6FA);
}
  
.cp-progress-main {
  display: flex;
}
.cp-progress-main .cp-progress-bg {
    width: 50px;
    background: #eaedf4;
    /* background: #ff6f00; */
    flex: 1;
    margin: 8px 0;
}
.cp-progress-main .cp-progress-bg  .cp-progress-bar {
    transition: width 1s;
}


</style>
