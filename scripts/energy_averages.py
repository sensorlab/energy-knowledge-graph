def monthly_plot(data: pd.DataFrame, key:str, month:int, year:int, ax, vmin:float=None, vmax:float=None):
    data['ds'] =  pd.to_datetime(data['ds'], infer_datetime_format=True)
    data = data.set_index('ds')

    data = data[[key]].resample('1H').mean()
    
    new_index = pd.date_range(start=f'{data.index.year.min()}-01-01T00:00', end=f'{data.index.year.max()+1}-01-01T00:00', freq='1H')
    data = data.reindex(new_index)

    data = data[(data.index.year == year) & (data.index.month == month)]

    day = data.index.day
    series = data[key]
    series = series.values.reshape(24, len(day.unique()), order="F")
    xgrid = np.arange(day.max() + 1) + 1
    ygrid = np.arange(25)
    
    ax.grid(False)

    ax.pcolormesh(xgrid, ygrid, series, cmap="magma", vmin=vmin, vmax=vmax)
    # Invert the vertical axis
    ax.set_ylim(24, 0)
    # Set tick positions for both axes
    ax.yaxis.set_ticks([i for i in range(24)])
    ax.xaxis.set_ticks([10, 20, 30])
    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(which='both', length=0)
    ax.xaxis.set_tick_params(which='both', length=0)
    
    # Remove all spines
    ax.set_frame_on(False)


def weekly_plot(data: pd.DataFrame, key: str, year:int, ax):
    # Fix this two lines to your needs. You must end up with DateTimeIndex and value with daily sampling
    data['ds'] =  pd.to_datetime(data['ds'], infer_datetime_format=True)
    data = data.set_index('ds')
    data = data[[key]].resample('1D').sum()

    new_index = pd.date_range(start=f'{data.index.year.min()}-01-01', end=f'{data.index.year.max()+1}-01-01', freq='1D')
    data = data.reindex(new_index)

    data = data[(data.index.year == year)]

    # Lets figure out the size of array
    n_weeks = data.index.isocalendar().week.unique().max() + 1
    n_days = 7

    # Sorry. I couldn't find more efficient way to do it.
    array = np.full(shape=(n_weeks, n_days), fill_value=np.nan)
    for ts, item in data.iterrows():
        array[ts.week, ts.dayofweek] = item[key]

    ax.imshow(array.T, cmap='magma', interpolation='none')

    ax.set_xlim(1, n_weeks)

    ax.grid(False)

    #ax.set_xticklabels(data.index.strftime('%b %d'), rotation=90)
    weekdays = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
    ax.set_yticks(range(7), weekdays)

    # ax.xaxis.set_ticks([10, 20, 30])
    # # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(which='both', length=0)
    ax.xaxis.set_tick_params(which='both', length=0)
    
    # # Remove all spines
    ax.set_frame_on(False)