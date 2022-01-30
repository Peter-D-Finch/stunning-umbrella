// https://fsymbols.com/generators/carty/
use crate::prediction_v2::*;

use crate::stats::*;
use crate::markov_chain::*;
use crate::technical_analysis::*;
use crate::data_mining::*;
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
			let round: (u64, u64, u64, u64, i64, i64, u64, u64, u64, u64, u64, u64, u64, bool) =
				contract.query("rounds",
							(epoch_prev-lookback+i,),
							None,
							Options::default(),
							None).await.unwrap();
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
	pub fn get_close_vec(data: &Vec<Prediction>) -> Vec<f64> {
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(data[i].close_price);
		}
		return result;
	}
	pub fn get_change_vec(data: &Vec<Prediction>) -> Vec<f64> {
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
	pub fn get_open_vec(data: &Vec<Prediction>) -> Vec<f64> {
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(data[i].lock_price);
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
	pub fn covariance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
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
	pub fn st_dev(data: &Vec<f64>) -> f64 {
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
		let cov = covariance(&a, &b);
		let a_stdev = st_dev(&a);
		let b_stdev = st_dev(&b);
		let correl = cov / (a_stdev * b_stdev);
		return correl;
	}
	pub fn z_score(x: f64, mean: f64, stdev: f64) -> f64 {
		return (x - mean) / stdev;
	}
	pub fn z_score_vec(data: Vec<f64>) -> Vec<f64> {
		let avg = mean(&data);
		let stdev = st_dev(&data);
		let mut result = Vec::new();
		for i in 0..data.len() {
			result.push(z_score(data[i], avg, stdev));
		}
		return result;
	}
	pub fn remove_outliers(data: Vec<f64>, n: f64, iter: i64) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		let z_scores = z_score_vec(data.clone());
		let stdev = st_dev(&data);
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
			let stdev_adj = st_dev(&result);
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
		println!("--------------------------------------------------------------------------------");
		Chart::new(150,80,-0.1,1.1).lineplot(&Shape::Points(points)).display();
		println!("--------------------------------------------------------------------------------");
	}
	pub fn plot_scat_dist(x: &Vec<f64>, y: &Vec<f64>) {
		let a = minmax_scale(&x);
		let b = minmax_scale(&y);
		let c = minmax_scale(&distribution(&x, &y));
		let points = &make_points(&a, &b)[..];
		let points2 = &make_points(&a, &c)[..];
		println!("--------------------------------------------------------------------------------");
		Chart::new(150,80,-0.1,1.1)
			.lineplot(&Shape::Points(points))
			.lineplot(&Shape::Points(points2))
			.display();
		println!("--------------------------------------------------------------------------------");
	}
	pub fn plot_timeseries(q: &Vec<f64>) {
		println!("-----------------------------------------------------------------------------------------------------");
		Chart::new(120,60,1.0,q.len() as f32).lineplot(&Shape::Continuous(Box::new(|x| q[(x as usize)%q.len()] as f32))).display();
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
	/*
	A module that implements a generic Markov Chain model.
	*/
	#![allow(dead_code)]
	use crate::stats::*;
	/*
	A function to convert any data in tabular form (2-D Vector) to a state-
	based system. A 2-D vector and number of states for each dimension is
	passed as arguments. Each value is rounded to an integer between
	0 and the number of states.
	
	@param data a 2-D Vector that contains the data
	@param states the number of states each dimension will have
	@return the resulting state-based system
	*/
	pub fn to_state_system(data: Vec<Vec<f64>>, states: usize) -> Vec<Vec<f64>> {
		let mut result: Vec<Vec<f64>> = Vec::new();
		for i in data.iter() {
			let mut temp: Vec<f64> = minmax_scale(&i);
			let mut _temp_int: Vec<i64> = Vec::new();
			for j in 0..temp.len() {
				temp[j] = temp[j] / (1.0/(states as f64));
				temp[j] = temp[j].floor();
				_temp_int.push(temp[j] as i64);
			}
			result.push(temp);
		}
		return result;
	}
	
	/*
	A function that converts a timeseries vector or states into either a 0 or 1
	based on whether it is above or below the median.
	
	@param data The vector of states
	@param states The number of states in the data parameter
	@return vector of 0s and 1s
	*/
	pub fn up_down(data: &Vec<f64>, states: usize) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		for i in data.iter() {
			if i > &((states / 2) as f64) { result.push(1.0); }
			else { result.push(0.0); }
		}
		return result;
	}
	
	/*
	A function that converts a timeseries vector or states into either a 0 or 1
	based on whether it is above or below zero.
	
	@param data The vector of states
	@param states The number of states in the data parameter
	@return vector of 0s and 1s
	*/
	pub fn ud_fuzzy(data: &Vec<f64>, median: f64) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		for i in data.iter() {
			if i > &median { result.push(1.0); }
			else { result.push(0.0); }
		}
		return result;
	}
	
	/*
	
	*/
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
	#![allow(dead_code)]
	/*
	A module that implements classical technical indicators from finance.
	*/
	use crate::stats::*;
	

//▀█▀ █▀█ █▀▀ █▄░█ █▀▄
//░█░ █▀▄ ██▄ █░▀█ █▄▀
	/* */
	pub fn moving_avg(data: &Vec<f64>, len: usize) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		for i in 0..data.len() {
			if i < len { result.push(mean(&data[0..i].to_vec())); }
			else {
				let temp: Vec<f64> = data[i-len .. i].to_vec();
				result.push(mean(&temp));
			}
		}
		return result;
	}
	
	/* */
	pub fn exp_moving_avg(data: &Vec<f64>, len: usize) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		for i in 0..data.len() {
			if i < len {
				if i == 0 { result.push(data[0]); }
				else { result.push(mean(&data[0..i].to_vec())); }
			}
			else {
				result.push( (data[i]*(2.0/(1.0+(len as f64)))) + (result[i-1]*(1.0-(2.0/(1.0+(len as f64))))) );
			}
		}
		return result;
	}
	

//█▀█ █▀ █▀▀ █ █░░ █░░ ▄▀█ ▀█▀ █▀█ █▀█ █▀
//█▄█ ▄█ █▄▄ █ █▄▄ █▄▄ █▀█ ░█░ █▄█ █▀▄ ▄█
	/* */
	pub fn macd(data: &Vec<f64>, len1: usize, len2: usize) -> Vec<f64> {
		let mut result: Vec<f64> = Vec::new();
		let ma1 = exp_moving_avg(&data, len1);
		let ma2 = exp_moving_avg(&data, len2);
		for i in 0..data.len() {
			result.push(ma1[i] - ma2[i]);
		}
		return result;
	}
	

//█░█ █▀█ █░░ ▄▀█ ▀█▀ █ █░░ █ ▀█▀ █▄█
//▀▄▀ █▄█ █▄▄ █▀█ ░█░ █ █▄▄ █ ░█░ ░█░


}
mod data_mining {
	#![allow(dead_code)]
	use crate::stats::*;
	use colour::*;
	
	
	//█▀▀ █░█ █▄░█ █▀▀ ▀█▀ █ █▀█ █▄░█ █▀
	//█▀░ █▄█ █░▀█ █▄▄ ░█░ █ █▄█ █░▀█ ▄█
	/* */
	fn sort_system(data_param: &Vec<Vec<f64>>, p: usize) -> Vec<Vec<f64>> {
		let mut data = data_param.clone();
		loop {
			let mut swapped = false;
			let mut i = 0;
			while i < data[p].len() {
				if i >= data[p].len()-1 { break; }
				if data[p][i] > data[p][i+1] {
					swapped = true;
					for j in 0..data.len() {
						let value = data[j][i];
						data[j].remove(i);
						data[j].insert(i+1, value);
					}
					break;
				}
				i += 1;
			}
			if !swapped { break; }
		}

		return data;
	}
	
	
	//█▀▄▀█ ▄▀█ ▀█▀ █▀█ █ █▀▀ █▀▀ █▀
	//█░▀░█ █▀█ ░█░ █▀▄ █ █▄▄ ██▄ ▄█
	/* */
	pub fn c_matrix(dataframe: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
		let mut result: Vec<Vec<f64>> = Vec::new();
		for i in 0..dataframe.len() {
			let mut temp: Vec<f64> = Vec::new();
			for j in dataframe.iter() {
				temp.push(pearson(&dataframe[i], &j));
			}
			result.push(temp);
		}
		return result;
	}
	
	/* */
	pub fn cov_matrix(dataframe: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
		let mut result: Vec<Vec<f64>> = Vec::new();
		for i in 0..dataframe.len() {
			let mut temp: Vec<f64> = Vec::new();
			for j in dataframe.iter() {
				temp.push(covariance(&dataframe[i], &j));
			}
			result.push(temp);
		}
		return result;
	}
	
	/* */
	pub fn c_matrix_mean(dataframe: &Vec<Vec<f64>>, num: usize) -> Vec<Vec<f64>> {
		let mut result: Vec<Vec<f64>> = Vec::new();
		for j in 0..dataframe.len() {
			let sort = sort_system(dataframe, j);
			let mut avg_state: Vec<f64> = Vec::new();
			for i in 0..sort.len() {
				avg_state.push(mean(&sort[i][sort[i].len()-num-1..].to_vec()));
			}
			result.push(avg_state);
		}
		return result;
	}
	
	/* */
	pub fn c_matrix_stdev(dataframe: &Vec<Vec<f64>>, num: usize) -> Vec<Vec<f64>> {
		let mut result: Vec<Vec<f64>> = Vec::new();
		for j in 0..dataframe.len() {
			let sort = sort_system(dataframe, j);
			let mut avg_state: Vec<f64> = Vec::new();
			for i in 0..sort.len() {
				avg_state.push(st_dev(&sort[i][sort[i].len()-num-1..].to_vec()));
			}
			
			result.push(avg_state);
		}
		return result;
	}
	
	
	//█▀▄ █ █▀ █▀█ █░░ ▄▀█ █▄█
	//█▄▀ █ ▄█ █▀▀ █▄▄ █▀█ ░█░
	/* */
	pub fn print_cov_matrix(dataframe: &Vec<Vec<f64>>, names: &Vec<&str>) {
		assert!(dataframe.len() == names.len());
		let matrix = cov_matrix(dataframe);
		print!("\t\t");
		for i in names.iter() { print!("{0}\t\t", i); }
		print!("\n");
		for i in 0..names.len() {
			white!("{0}\t\t", names[i]);
			for j in matrix[i].iter() {
				if j < &0.0 { red!("{0:.6}\t", j); }
				else if j > &0.0 { green!("{0:.6}\t", j); }
				
			}
			print!("\n");
		}
	}
	
	/* */
	pub fn print_cor_matrix(dataframe: &Vec<Vec<f64>>, names: &Vec<&str>) {
		assert!(dataframe.len() == names.len());
		let matrix = c_matrix(dataframe);
		print!("\t\t");
		for i in names.iter() { print!("{0}\t\t", i); }
		print!("\n");
		for i in 0..names.len() {
			white!("{0}\t\t", names[i]);
			for j in matrix[i].iter() {
				if j < &0.3 { white!("{0:.6}\t", j); }
				else if j.abs() < (0.5 as f64) { red!("{0:.6}\t", j); }
				else if j.abs() < (0.7 as f64) { yellow!("{0:.6}\t", j); }
				else if j.abs() > (0.7 as f64) { green!("{0:.6}\t", j); }
			}
			print!("\n");
		}
	}
	
	/* */
	pub fn print_c_matrix_mean(dataframe: &Vec<Vec<f64>>, names: &Vec<&str>, num: usize, dim: usize) {
		assert!(dataframe.len() == names.len());
		let matrix = c_matrix_mean(dataframe, num);
		print!("\t\t");
		for i in names.iter() { print!("Avg. {0}\t", i); }
		print!("\n");
		for i in 0..names.len() {
			white!("Top[{0}] {1}\t", num, names[i]);
			for j in 0..matrix[i].len() {
				let temp = matrix[i][j]/(dim as f64);
				if i == j { green!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
				else if temp < 0.25 { red!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
				else if temp > 0.75 { green!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
				else if temp > 0.625 { yellow!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
				else if temp < 0.375 { yellow!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
				else { white!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
			}
			print!("\n");
		}
	}
	
	/* */
	pub fn print_c_matrix_stdev(dataframe: &Vec<Vec<f64>>, names: &Vec<&str>, num: usize, dim: usize) {
		assert!(dataframe.len() == names.len());
		let matrix = c_matrix_stdev(dataframe, num);
		print!("\t\t");
		for i in names.iter() { print!("Avg. {0}\t", i); }
		print!("\n");
		for i in 0..names.len() {
			white!("Top[{0}] {1}\t", num, names[i]);
			for j in 0..matrix[i].len() {
				if i == j { green!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
				else { white!("[{0:.4}]\t", matrix[i][j]/(dim as f64)); }
			}
			print!("\n");
		}
	}
	
	/* */
	pub fn display_analysis(dataframe: &Vec<Vec<f64>>, names: &Vec<&str>, dim: usize) {
		println!("\n\nCOVARIANCE MATRIX");
		print_cov_matrix(dataframe, names);
		println!("\n\nCORRELATION MATRIX");
		print_cor_matrix(dataframe, names);
		println!("\n\nAVERAGE STATE CLUSTER MATRIX");
		print_c_matrix_mean(dataframe, names, 10, dim);
		println!("\n\nSTDEV STATE CLUSTER MATRIX");
		print_c_matrix_stdev(dataframe, names, 10, dim);
		println!("\n\nEND");
	}
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	
	// Constants
	let num_data = 1000;
	let dimensions: usize = 50;
	
	// Collect Data
	let raw_data = join!(historical_rounds(num_data)).0.unwrap();
	let a: Vec<f64> = remove_outliers(get_change_vec(&raw_data), 4.0, 30);
	let b: Vec<f64> = remove_outliers(get_bbr_vec(&raw_data), 4.0, 30);
	let c: Vec<f64> = get_close_vec(&raw_data);
	let d: Vec<f64> = macd(&get_open_vec(&raw_data), 16, 24);
	let e: Vec<f64> = remove_outliers(get_pool_vec(&raw_data), 4.0, 30);
	
	// Functions
	let mut data: Vec<Vec<f64>> = vec![a.clone(), b.clone(), c.clone(), d.clone(), e.clone()];
	data = to_state_system(data, dimensions.clone());
	
	// Display
	display_analysis(&data, &(vec!["%CHG", "B2BR", "CPRC", "MACD", "POOL"]), dimensions.clone());
	
	Ok(())
}