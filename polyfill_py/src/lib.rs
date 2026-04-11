/*!
PyO3 Python bindings for polyfill-rs.

Exposes `PolyRustClient` — a drop-in replacement for the Python
`PolymarketClient` that runs the hot paths (market fetching, order book
queries, order submission) through the polyfill-rs Rust client.

All methods are *synchronous* at the Rust/Python boundary; callers should
dispatch them via `loop.run_in_executor(None, ...)` just like the existing
Python client.

Build:
    cd polyfill_py && maturin build --release
    pip install target/wheels/polyfill_py-*.whl

Then in Python:
    from polyfill_py import PolyRustClient
*/

use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rust_decimal::Decimal;
use serde_json::json;
use tokio::runtime::Runtime;

use polyfill_rs::{
    ApiCredentials, ClobClient, MidpointResponse, OrderArgs, OrderType,
    OrderBookSummary, Side, SpreadResponse,
};
// MarketOrderArgs lives in the types sub-module (not re-exported at crate root)
use polyfill_rs::types::MarketOrderArgs;

// One shared Tokio runtime for all blocking calls.
static RT: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("polyfill_py: failed to create Tokio runtime")
});

fn to_py(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// PolyRustClient
// ---------------------------------------------------------------------------

/// High-performance Polymarket CLOB client backed by polyfill-rs.
///
/// Mirrors the interface of `polymarket.client.PolymarketClient` so that
/// `polymarket/rust_bridge.py` can swap it in transparently.
#[pyclass]
struct PolyRustClient {
    inner: ClobClient,
}

#[pymethods]
impl PolyRustClient {
    /// Create and authenticate a client.
    ///
    /// Parameters
    /// ----------
    /// host : str
    ///     CLOB host URL, e.g. ``"https://clob.polymarket.com"``.
    /// private_key : str
    ///     Hex-encoded Ethereum private key (without 0x prefix).
    /// chain_id : int
    ///     137 for Polygon mainnet (default).
    /// api_key, api_secret, api_passphrase : str | None
    ///     L2 API credentials.  When all three are supplied the client uses
    ///     L2 auth (signed HMAC headers).  Omit for read-only market data.
    #[new]
    #[pyo3(signature = (
        host,
        private_key,
        chain_id = 137,
        api_key = None,
        api_secret = None,
        api_passphrase = None,
    ))]
    fn new(
        host: &str,
        private_key: &str,
        chain_id: u64,
        api_key: Option<&str>,
        api_secret: Option<&str>,
        api_passphrase: Option<&str>,
    ) -> PyResult<Self> {
        let inner = match (api_key, api_secret, api_passphrase) {
            (Some(k), Some(s), Some(p)) => {
                let creds = ApiCredentials {
                    api_key:    k.to_string(),
                    secret:     s.to_string(),
                    passphrase: p.to_string(),
                };
                ClobClient::with_l2_headers(host, private_key, chain_id, creds)
            }
            _ => ClobClient::with_l1_headers(host, private_key, chain_id),
        };

        Ok(PolyRustClient { inner })
    }

    // -----------------------------------------------------------------------
    // Account
    // -----------------------------------------------------------------------

    /// Return the Ethereum address derived from the private key, or None.
    fn get_address(&self) -> Option<String> {
        self.inner.get_address()
    }

    // -----------------------------------------------------------------------
    // Market data
    // -----------------------------------------------------------------------

    /// Fetch a page of markets.  Returns JSON string.
    #[pyo3(signature = (next_cursor = None))]
    fn get_markets(&self, next_cursor: Option<&str>) -> PyResult<String> {
        let resp = RT
            .block_on(self.inner.get_markets(next_cursor))
            .map_err(to_py)?;
        serde_json::to_string(&resp).map_err(to_py)
    }

    /// Fetch the mid-price for a single token (0–1 scale).
    fn get_midpoint(&self, token_id: &str) -> PyResult<f64> {
        let resp: MidpointResponse = RT
            .block_on(self.inner.get_midpoint(token_id))
            .map_err(to_py)?;
        resp.mid
            .to_string()
            .parse::<f64>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Fetch the full order book for a token.  Returns JSON string.
    fn get_order_book(&self, token_id: &str) -> PyResult<String> {
        let resp: OrderBookSummary = RT
            .block_on(self.inner.get_order_book(token_id))
            .map_err(to_py)?;
        // OrderBookSummary doesn't derive Serialize; build JSON manually.
        let val = json!({
            "market":    resp.market,
            "asset_id":  resp.asset_id,
            "timestamp": resp.timestamp,
            "bids": resp.bids.iter().map(|b| json!({
                "price": b.price.to_string(),
                "size":  b.size.to_string(),
            })).collect::<Vec<_>>(),
            "asks": resp.asks.iter().map(|a| json!({
                "price": a.price.to_string(),
                "size":  a.size.to_string(),
            })).collect::<Vec<_>>(),
        });
        serde_json::to_string(&val).map_err(to_py)
    }

    /// Fetch the spread for a token.  Returns JSON string ``{"spread": "0.02"}``.
    fn get_spread(&self, token_id: &str) -> PyResult<String> {
        let resp: SpreadResponse = RT
            .block_on(self.inner.get_spread(token_id))
            .map_err(to_py)?;
        // SpreadResponse only has `spread`; build JSON manually.
        let val = json!({ "spread": resp.spread.to_string() });
        serde_json::to_string(&val).map_err(to_py)
    }

    // -----------------------------------------------------------------------
    // Sampling / simplified markets (for startup seeding — smaller payload)
    // -----------------------------------------------------------------------

    /// Fetch sampling markets.  Returns JSON string.
    #[pyo3(signature = (next_cursor = None))]
    fn get_sampling_markets(&self, next_cursor: Option<&str>) -> PyResult<String> {
        let resp = RT
            .block_on(self.inner.get_sampling_markets(next_cursor))
            .map_err(to_py)?;
        serde_json::to_string(&resp).map_err(to_py)
    }

    // -----------------------------------------------------------------------
    // Balances
    // -----------------------------------------------------------------------

    /// Return the USDC balance in the trading wallet (in USDC, not wei).
    ///
    /// Reads the L2 balance-allowance endpoint.  USDC on Polygon has 6 decimals.
    fn get_usdc_balance(&self) -> PyResult<f64> {
        let resp = RT
            .block_on(self.inner.get_balance_allowance(None))
            .map_err(to_py)?;

        // resp is a serde_json::Value — extract "balance" field (micro-USDC string)
        let raw = resp
            .get("balance")
            .and_then(|v| v.as_str())
            .unwrap_or("0");
        raw.parse::<f64>()
            .map(|v| v / 1_000_000.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Order placement
    // -----------------------------------------------------------------------

    /// Place a FOK market BUY order.
    ///
    /// Note: polyfill-rs `create_market_order` implies BUY side.
    /// For SELL orders use `create_limit_order` with a price at or below mid.
    ///
    /// Returns JSON string of the PostOrderResponse.
    fn create_market_buy(
        &self,
        token_id: &str,
        amount_usdc: f64,
    ) -> PyResult<String> {
        let amount = Decimal::try_from(amount_usdc).map_err(to_py)?;

        let args = MarketOrderArgs {
            token_id: token_id.to_string(),
            amount,
        };

        let signed = RT
            .block_on(self.inner.create_market_order(&args, None, None))
            .map_err(to_py)?;

        let resp = RT
            .block_on(self.inner.post_order(signed, OrderType::FOK))
            .map_err(to_py)?;

        serde_json::to_string(&resp).map_err(to_py)
    }

    /// Place a GTC limit order (BUY or SELL).
    ///
    /// Parameters
    /// ----------
    /// token_id : str
    /// side : str
    ///     "BUY" or "SELL".
    /// price : float
    ///     0–1 scale (e.g. 0.55 = 55 cents).
    /// size : float
    ///     Number of shares.
    ///
    /// Returns JSON string of the PostOrderResponse.
    fn create_limit_order(
        &self,
        token_id: &str,
        side: &str,
        price: f64,
        size: f64,
    ) -> PyResult<String> {
        let side  = parse_side(side)?;
        let price = Decimal::try_from(price).map_err(to_py)?;
        let size  = Decimal::try_from(size).map_err(to_py)?;

        // polyfill_rs::OrderArgs (re-exported from client) = { token_id, price, size, side }
        let args = OrderArgs {
            token_id: token_id.to_string(),
            side,
            price,
            size,
        };

        // create_and_post_order returns serde_json::Value directly
        let resp = RT
            .block_on(self.inner.create_and_post_order(&args))
            .map_err(to_py)?;

        serde_json::to_string(&resp).map_err(to_py)
    }

    // -----------------------------------------------------------------------
    // Connection warmup
    // -----------------------------------------------------------------------

    /// Pre-warm HTTP/2 connections.  Call once after construction.
    fn prewarm(&self) -> PyResult<()> {
        RT.block_on(self.inner.prewarm_connections()).map_err(to_py)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_side(s: &str) -> PyResult<Side> {
    match s.to_uppercase().as_str() {
        "BUY"  => Ok(Side::BUY),
        "SELL" => Ok(Side::SELL),
        other  => Err(PyRuntimeError::new_err(format!("invalid side: '{other}'"))),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn polyfill_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolyRustClient>()?;
    Ok(())
}
