use crate::prediction_v2::*;
use crate::stats_plot::*;
use crate::stats::*;
use crate::markov_chain::*;
use futures::join;

mod prediction_v2 {
	#![allow(dead_code)]
	use web3::contract::Contract;
	use web3::contract::Options;
	use web3::types::H160;
	use std::str::FromStr;
	#[derive(Clone)]
	pub struct Prediction {
		pub lock_price: f64,
		pub close_price: f64,
		pub price_change: f64,
		pub bull_amount: f64,
		pub bear_amount: f64,
		pub bb_ratio: f64,
		pub pool_amount: f64
	}
	pub async fn new_prediction(epoch: u64) -> Result<Prediction, Box<dyn std::error::Error>> {
		let transport = web3::transports::Http::new("https://bsc-dataseed1.binance.org:443")?;
		let web3 = web3::Web3::new(transport);
		let contract = Contract::from_json(web3.eth(), H160::from_str("0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA").unwrap(), include_bytes!("../ABI/token.json"))?;
		let round: (u64, u64, u64, u64, i64, i64, u64, u64, u64, u64, u64, u64, u64, bool) = contract.query("rounds", (epoch,), None, Options::default(), None).await.unwrap();
		let pc: f64 = (round.4 as f64) - (round.5 as f64);
		let bb: f64 = (round.9 as f64) / (round.10 as f64);
		let result: Prediction = Prediction {
			lock_price: round.4 as f64,
			close_price: round.5 as f64,
			price_change: pc,
			bull_amount: round.9 as f64,
			bear_amount: round.10 as f64,
			bb_ratio: bb,
			pool_amount: round.8 as f64
		};
		Ok(result)
	}
	pub async fn historical_rounds(lookback: u64) -> Result<Vec<Prediction>, Box<dyn std::error::Error>> {
		let transport = web3::transports::Http::new("https://bsc-dataseed1.binance.org:443")?;
		let web3 = web3::Web3::new(transport);
		let contract = Contract::from_json(web3.eth(), H160::from_str("0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA").unwrap(), include_bytes!("../ABI/token.json"))?;
		let mut data: Vec<Prediction> = Vec::new();
		let epoch_now: u64 = contract.query("currentEpoch", (), None, Options::default(), None).await.unwrap();
		let epoch_prev: u64 = epoch_now - 1;
		for i in 0..lookback {
			let round: (u64, u64, u64, u64, i64, i64, u64, u64, u64, u64, u64, u64, u64, bool) = contract.query("rounds", (epoch_prev-lookback+i,), None, Options::default(), None).await.unwrap();
			let pc: f64 = (round.4 as f64) - (round.5 as f64);
			let bb: f64 = (round.9 as f64) / (round.10 as f64);
			let result: Prediction = Prediction {
				lock_price: round.4 as f64,
				close_price: round.5 as f64,
				price_change: pc,
				bull_amount: round.9 as f64,
				bear_amount: round.10 as f64,
				bb_ratio: bb,
				pool_amount: round.8 as f64
			};
			data.push(result);
		}
		Ok(data)
	}
	pub fn get_bbr_vec(data: &Vec<Prediction>) -> Vec<f64> {
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(data[i].bb_ratio);
		}
		return result;
	}
	pub fn get_close_vec(data: Vec<Prediction>) -> Vec<f64> {
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(data[i].close_price);
		}
		return result;
	}
	pub fn get_change_vec(data: Vec<Prediction>) -> Vec<f64> {
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(data[i].price_change);
		}
		return result;
	}
	pub fn get_pool_vec(data: &Vec<Prediction>) -> Vec<f64> {
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(data[i].pool_amount);
		}
		return result;
	}
}
mod pswap {
	use serde_json::Value;
	use serde_json::Map;
	use async_trait::async_trait;
 
	pub struct Asset {
		pub price: f64,
		pub liquidity: f64,
		pub quote_volume: f64,
		pub base_volume: f64,
		pub address: String
	}
	#[async_trait]
	pub trait WebBehavior {
		async fn new(addr: String) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized;
		/*async fn update(&mut self) -> Result<(), Box<dyn std::error::Error>>;*/
		fn print(&self);
	}
	#[async_trait]
	impl WebBehavior for Asset {
		async fn new(addr: String) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized {
			let client = reqwest::Client::builder().build()?;
			let result = client.get("https://api.pancakeswap.info/api/v2/summary").send().await?.text().await?;
			let json: Map<String, Value> = serde_json::from_str(&result)?;
			let item = json.get("data").unwrap().get(addr.clone()).unwrap();
			let prc = item.get("price").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			let liq = item.get("liquidity").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			let bvol = item.get("base_volume").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			let qvol = item.get("quote_volume").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			let result: Asset = Asset {price: prc, liquidity: liq, quote_volume: qvol, base_volume: bvol, address: addr};
			Ok(result)
		}
		/*async fn update(&mut self) -> Result<(), Box<dyn std::error::Error>> {
			let client = reqwest::Client::builder().build()?;
			let result = client.get("https://api.pancakeswap.info/api/v2/summary").send().await?.text().await?;
			let json: Map<String, Value> = serde_json::from_str(&result)?;
			let item = json.get("data").unwrap().get(self.address.clone()).unwrap();
			self.price = item.get("price").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			self.liquidity = item.get("liquidity").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			self.base_volume = item.get("base_volume").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			self.quote_volume = item.get("quote_volume").unwrap().as_str().unwrap().parse::<f64>().unwrap();
			Ok(())
		}*/
		fn print(&self) {
			println!("{0}, {1}, {2}, {3}", self.price, self.liquidity, self.base_volume, self.quote_volume);
		}
	}
}
mod stats {
	/*
	TODO:
	 - Spearman Rank
	 - rolling_stats
	 - Approximate Entropy
	*/
	#![allow(dead_code)]
	use fitting::gaussian::val;
	use fitting::gaussian::fit;
	use ndarray::Array;
	use ndarray::Dim;
	pub fn mean(data: &Vec<f64>) -> f64 {
		let mut sum = 0.0;
		for i in 0..data.len() {
			sum = sum + data[i];
		}
		let avg = sum / (data.len() as f64);
		return avg;
	}
	pub fn covariance(a: Vec<f64>, b: Vec<f64>) -> f64 {
		assert!(a.len() == b.len());
		let a_avg = mean(&a);
		let b_avg = mean(&b);
		let mut sum = 0.0;
		for i in 0..a.len() {
			sum = sum + ((a[i] - a_avg)*(b[i] - b_avg));
		}
		let covariance = sum / (a.len() as f64);
		return covariance;
	}
	pub fn st_dev(data: Vec<f64>) -> f64 {
		let avg = mean(&data);
		let mut diff: Vec<f64> = Vec::new();
		for i in 0..data.len() {
			diff.push(f64::powf(data[i] - avg, 2.0));
		}
		let avg_diff = mean(&diff);
		let st_dev: f64 = avg_diff.sqrt();
		return st_dev;
	}
	pub fn pearson(a: &Vec<f64>, b: &Vec<f64>) -> f64{
		assert!(a.len() == b.len());
		let cov = covariance(a.clone(), b.clone());
		let a_stdev = st_dev(a.clone());
		let b_stdev = st_dev(b.clone());
		let correl = cov / (a_stdev * b_stdev);
		return correl;
	}
	pub fn z_score(x: f64, mean: f64, stdev: f64) -> f64 {
		return (x - mean) / stdev;
	}
	pub fn z_score_vec(data: Vec<f64>) -> Vec<f64> {
		let avg = mean(&data);
		let stdev = st_dev(data.clone());
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(z_score(data[i], avg, stdev));
		}
		return result;
	}
	pub fn remove_outliers(data: Vec<f64>, n: f64, iter: i64) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		let z_scores = z_score_vec(data.clone());
		let stdev = st_dev(data.clone());
		let avg = mean(&data);
		for i in 0..z_scores.len() {
			if z_scores[i] > n {
				result.push(avg + (stdev * n));
			}
			else if z_scores[i] < (n * -1.0) {
				result.push(avg - (stdev * n));
			}
			else {
				result.push(data[i]);
			}
		}
		for _j in 0..iter {
			let z_scores_adj = z_score_vec(result.clone());
			let stdev_adj = st_dev(result.clone());
			let avg_adj = mean(&result);
			for i in 0..z_scores_adj.len() {
				if z_scores_adj[i] > n {
					result[i] = avg_adj + (stdev_adj * n);
				}
				else if z_scores_adj[i] < (n * -1.0) {
					result[i] = avg_adj - (stdev_adj * n);
				}
			}
		}
		return result;
	}
	pub fn minmax_scale(data: &Vec<f64>) -> Vec<f64> {
		let min = data.iter().fold(0.0/0.0, |m, v| v.min(m));
		let max = data.iter().fold(0.0/0.0, |m, v| v.max(m));
		let mut result: Vec<f64> = Vec::new();
		for i in 0..data.len() {
			result.push((data[i] - min) / (max - min));
		}
		return result;
	}
	pub fn distribution(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
		let a2: ndarray::Array<f64, Dim<[usize; 1]>> = Array::from_iter(a.clone());
		let b2: ndarray::Array<f64, Dim<[usize; 1]>> = Array::from_iter(b.clone());
		let gauss: (f64, f64, f64) = fit(a2.clone(), b2.clone()).expect("bleh");
		let mut c: Vec<f64> = Vec::new();
		for i in 0..a.len() {
			c.push(val(a2[i], gauss.0, gauss.1, gauss.2));
		}
		return c;
	}
	pub fn derivative(data: &Vec<f64>, ord: i64) -> Vec<f64> {
		let mut result: Vec<f64> = vec![0.0];
		for i in 1..data.len() {
			result.push(data[i] - data[i-1]);
		}
		result[0] = result[1];
		for _j in 0..ord {
			let temp: Vec<f64> = result.clone();
			for i in 1..data.len() {
				result[i] = temp[i] - temp[i-1];
			}
			result[0] = result[1];
		}
		return result;
	}
	fn make_points(a: &Vec<f64>, b: &Vec<f64>) -> Vec<(f64, f64)> {
		assert!(a.len() == b.len());
		let mut result: Vec<(f64, f64)> = Vec::new();
		for i in 0..a.len() {
			result.push((a[i], b[i]));
		}
		return result;
	}
}
mod stats_plot {
	#![allow(dead_code)]
	use textplots::{Chart, Shape, Plot};
	use crate::stats::minmax_scale;
	use crate::stats::distribution;
	pub fn plot_scatter(x: &Vec<f64>, y: &Vec<f64>) {
		let a = minmax_scale(&x);
		let b = minmax_scale(&y);
		let points = &make_points(&a, &b)[..];
		println!("-----------------------------------------------------------------------------------------------------");
		Chart::new(180,120,-0.1,1.1).lineplot(&Shape::Points(points)).display();
		println!("-----------------------------------------------------------------------------------------------------");
	}
	pub fn plot_scat_dist(x: &Vec<f64>, y: &Vec<f64>) {
		let a = minmax_scale(&x);
		let b = minmax_scale(&y);
		let c = minmax_scale(&distribution(&x, &y));
		let points = &make_points(&a, &b)[..];
		let points2 = &make_points(&a, &c)[..];
		println!("-----------------------------------------------------------------------------------------------------");
		Chart::new(180,120,-0.1,1.1)
			.lineplot(&Shape::Points(points))
			.lineplot(&Shape::Points(points2))
			.display();
		println!("-----------------------------------------------------------------------------------------------------");
	}
	pub fn plot_timeseries(q: &Vec<f64>) {
		println!("-----------------------------------------------------------------------------------------------------");
		Chart::new(180,60,1.0,q.len() as f32).lineplot(&Shape::Continuous(Box::new(|x| q[(x as usize)%q.len()] as f32))).display();
		println!("-----------------------------------------------------------------------------------------------------");
	}
	fn make_points(a: &Vec<f64>, b: &Vec<f64>) -> Vec<(f32, f32)> {
		assert!(a.len() == b.len());
		let mut result: Vec<(f32, f32)> = Vec::new();
		for i in 0..a.len() {
			result.push((a[i] as f32, b[i] as f32));
		}
		return result;
	}
}
mod markov_chain {
	#![allow(dead_code)]
	use crate::stats::*;
	pub fn to_state_system(data: Vec<Vec<f64>>, dimensions: i64) -> Vec<Vec<f64>> {
		let mut result: Vec<Vec<f64>> = Vec::new();
		for i in data.iter() {
			let mut temp: Vec<f64> = minmax_scale(&i);
			let mut _temp_int: Vec<i64> = Vec::new();
			for j in 0..temp.len() {
				temp[j] = temp[j] / (1.0/(dimensions as f64));
				temp[j] = temp[j].floor();
				_temp_int.push(temp[j] as i64);
			}
			result.push(temp);
		}
		return result;
	}
	pub fn binary_convert(data: &Vec<f64>, states: i64) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		for i in data.iter() {
			if i > &((states / 2) as f64) {
				result.push(1.0);
			}
			else {
				result.push(0.0);
			}
		}
		return result;
	}
	pub fn and_gate(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
		assert!(a.len() == b.len());
		let mut result: Vec<f64> = Vec::new();
		for i in 0..a.len() {
			if a[i] == b[i] {
				result.push(1.0);
			}
			else {
				result.push(0.0);
			}
		}
		return result;
	}
}
mod technical_analysis {
	
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	
	let dimensions = 50;
	let num_data = 1000;
	
	let raw_data = join!(historical_rounds(num_data)).0.unwrap();
	let a: Vec<f64> = remove_outliers(get_change_vec(raw_data.clone()), 4.0, 30);
	let b: Vec<f64> = remove_outliers(get_bbr_vec(&raw_data), 4.0, 30);
	let mut data: Vec<Vec<f64>> = vec![a.clone(), b.clone()];
	data = to_state_system(data, dimensions);
	data[0] = binary_convert(&data[0], dimensions);
	data[1] = binary_convert(&data[1], dimensions);
	plot_scat_dist(&a, &b);
	println!("avg: {0}", mean(&and_gate(&data[0], &data[1])));
	Ok(())
}